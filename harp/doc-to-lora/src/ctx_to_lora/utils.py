import ast
import gc
import hashlib
import logging
import os
import random
import re
import string
import time
from collections.abc import Iterable
from contextlib import contextmanager
from enum import Enum

import torch
import yaml
from peft import PeftConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import get_peft_model_state_dict

TRAINING_TASK = Enum("TRAINING_TASK", ["CAUSAL_LM", "COMPLETION"])


logger = logging.getLogger()


def shape_aware_target_type(module_name: str, module: torch.nn.Module) -> str:
    bare = module_name.split(".")[-1]
    return f"{bare}__in{int(module.in_features)}__out{int(module.out_features)}"


def peft_module_target_type(module_name: str, module: torch.nn.Module) -> str:
    base_layer = module.base_layer if hasattr(module, "base_layer") else module
    return shape_aware_target_type(module_name, base_layer)


def layer_index_from_module_name(module_name: str) -> int:
    match = re.search(r"\bmodel\.(?:language_model\.)?layers\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    match = re.search(r"\blayers\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse layer index from module name: {module_name}")


def filter_target_module_names(
    model: torch.nn.Module,
    target_modules: list[str],
    module_name_regex: str | None = None,
) -> list[str]:
    name_re = re.compile(module_name_regex) if module_name_regex else None
    out = []
    for name, module in model.named_modules():
        if name_re is not None and not name_re.search(name):
            continue
        if name.split(".")[-1] not in target_modules:
            continue
        if isinstance(module, torch.nn.Linear):
            out.append(name)
    return sorted(out)


# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
@contextmanager
def evaluating(*models):
    """Temporarily switch to evaluation mode."""
    is_training = [model.training if model is not None else False for model in models]
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield models
    finally:
        for model, training in zip(models, is_training):
            if model is not None:
                model.train(training)


def get_layers(model):
    if hasattr(model, "language_model"):
        return get_layers(model.language_model)
    if hasattr(model, "model"):
        return get_layers(model.model)
    return model.layers


def set_layers(model, layers):
    if hasattr(model, "language_model"):
        return set_layers(model.language_model, layers)
    if hasattr(model, "model"):
        return set_layers(model.model, layers)
    model.layers = layers


def get_num_layers(model):
    return len(get_layers(model))


def get_base_model(model):
    if hasattr(model, "model"):
        return get_base_model(model.model)
    return model


def get_num_params(model):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    return total_params, trainable_params


def log_num_train_params(model):
    logger.debug("Trainable model parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.debug(f"{name}, dtype:{p.dtype}")

    num_total_params, num_trainable_params = get_num_params(model)
    logger.info(
        f"trainable params: {num_trainable_params:,d} "
        f"|| all params: {num_total_params:,d} "
        f"|| trainable%: {100 * num_trainable_params / num_total_params:.4f}"
    )


def get_run_name(seed_str: str | None = None):
    if not seed_str:
        uuid = "".join(
            [random.choice(string.ascii_letters + string.digits) for _ in range(8)]
        )
        run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
    else:
        # Generate a UUID from the seed string
        hash_object = hashlib.sha256(seed_str.encode())
        uuid = hash_object.hexdigest()[:8]  # Take the first 8 characters of the hash
        run_name = seed_str + f"_{uuid}"
    return run_name


def try_convert(s):
    try:
        return ast.literal_eval(s)
    except:
        return s


def extract_cli_args(argv: list[str]):
    out = dict()
    for elem in argv:
        if elem.endswith(".yaml"):
            out["config"] = elem

        elif elem.startswith("--"):
            k, v = elem.split("=")
            k = k.split("--")[1]
            v = try_convert(v)
            # if k.startswith('env_'):
            #     k = k.split('_')[1]
            out[k] = v
    return out


def setup_logging(output_dir, debug=False):
    global logger

    os.makedirs(output_dir, exist_ok=True)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_level = logging.DEBUG if debug else logging.INFO
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    log_path = f"{output_dir}/debug.log"
    debug_handler = logging.FileHandler(log_path, delay=True)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(log_formatter)
    logger.addHandler(debug_handler)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Logging to: {log_path}")


def validate_args(args_list):
    # there shouldn't be overlap between args
    keys = set()
    for args in args_list:
        logger.debug(args)
        args_keys = set(vars(args).keys())
        assert len(keys & args_keys) == 0, "Overlap between args"
        keys |= args_keys


def save_yaml(data, path):
    # Filter out non-primitive fields
    data = {
        k: v
        for k, v in data.items()
        if isinstance(v, (int, float, str, bool, list, dict, type(None)))
    }

    with open(path, "w") as file:
        yaml.dump(data, file)


def get_peft_modules(model: PeftModel, peft_config: PeftConfig) -> list[dict[str, str]]:
    module_name_regex = getattr(peft_config, "module_name_regex", None)
    name_re = re.compile(module_name_regex) if module_name_regex else None
    loose_name_re = (
        re.compile(module_name_regex.lstrip("^")) if module_name_regex else None
    )
    bare_targets = getattr(peft_config, "bare_target_modules", peft_config.target_modules)
    target_shapes = set(getattr(peft_config, "target_module_shapes", None) or [])

    def name_matches(name: str) -> bool:
        if name_re is None:
            return True
        return bool(name_re.search(name) or loose_name_re.search(name))

    modules = []
    for name, module in model.named_modules():
        if not (
            name_matches(name)
            and name.split(".")[-1] in bare_targets
            and isinstance(module, BaseTunerLayer)
            and check_target_module_exists(peft_config, name)
        ):
            continue
        if target_shapes and peft_module_target_type(name, module) not in target_shapes:
            continue
        modules.append({"name": name, "module": module})
    return modules


def get_peft_in_out_features(
    model: PeftModel,
    peft_config: PeftConfig | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    if peft_config is None:
        return None, None
    in_features = dict()
    out_features = dict()
    for module_info in get_peft_modules(model, peft_config):
        module_name = module_info["name"]
        module = module_info["module"]
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, torch.nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        name = peft_module_target_type(module_name, module)

        if name not in in_features:
            in_features[name] = module.in_features
            out_features[name] = module.out_features
        else:
            assert in_features[name] == module.in_features
            assert out_features[name] == module.out_features

    return in_features, out_features


def generated_lora_to_state_dict(
    lora_dict: dict,
    module_names: dict,
    target_modules: list[str],
    layer_indices: Iterable[int],
) -> dict:
    lora_state_dict = dict()
    for target_module in target_modules:
        for local_idx, layer_idx in enumerate(layer_indices):
            for module_name in module_names[target_module][local_idx]:
                if "lora_A" in module_name:
                    lora_state_dict[module_name] = (
                        lora_dict[target_module]["A"][local_idx].cpu().contiguous()
                    )
                elif "lora_B" in module_name:
                    lora_state_dict[module_name] = (
                        lora_dict[target_module]["B"][local_idx].cpu().contiguous()
                    )
                else:
                    raise ValueError(f"Unexpected module name: {module_name}")
    return lora_state_dict


def get_lora_module_names(
    model: PeftModel,
    target_modules: list[str],
    layer_indices: Iterable[int],
) -> dict[str, list[str]]:
    layer_indices = [int(i) for i in layer_indices]
    layer_to_local = {layer_idx: i for i, layer_idx in enumerate(layer_indices)}
    module_names = {
        target_module: [[] for _ in range(len(layer_indices))]
        for target_module in target_modules
    }
    for k in get_peft_model_state_dict(model):
        if "lora" not in k:
            continue
        layer_idx = layer_index_from_module_name(k)
        if layer_idx in layer_to_local:
            for target_module in target_modules:
                bare_target = target_module.split("__in", maxsplit=1)[0]
                if f".{bare_target}." in k:
                    module_names[target_module][layer_to_local[layer_idx]].append(k)
                    break
    return module_names


def compile_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.compile()


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()


def concat_list(l):
    out = []
    for x in l:
        out += x
    return out


def check_is_iterable(x):
    try:
        iter(x)
    except TypeError:
        return False
    return True
