import logging
import os

import torch
from peft import PeftModel
from peft import get_peft_config as _get_peft_config
from peft.utils import PeftType
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)


logger = logging.getLogger()

GEMMA_VISION_MODELS = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]


def check_is_vision_model(model_name):
    return model_name in GEMMA_VISION_MODELS


def get_model_and_tokenizer(
    model_name_or_path,
    train,
    requires_grad,
    model_revision=None,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    tokenizer_kwargs=None,
    use_q_lora=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    model_kwargs = dict(model_kwargs or {})
    tokenizer_kwargs = dict(tokenizer_kwargs or {})
    if model_revision is not None:
        model_kwargs.setdefault("revision", model_revision)
        tokenizer_kwargs.setdefault("revision", model_revision)
    model = get_model(
        model_name_or_path,
        train,
        requires_grad,
        use_flash_attn,
        peft_config,
        model_kwargs,
        use_q_lora,
        device,
        dtype,
    )
    tokenizer = get_tokenizer(model_name_or_path, tokenizer_kwargs, peft_config, train)
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_tokenizer(
    model_name_or_path, tokenizer_kwargs=None, peft_config=None, train=False
):
    padding_side = "left" if not train else "right"
    truncation_side = "left"

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    local_files_only = os.getenv("HF_LOCAL_FILES_ONLY") or os.getenv("TRANSFORMERS_OFFLINE")
    if local_files_only and "local_files_only" not in tokenizer_kwargs:
        tokenizer_kwargs["local_files_only"] = local_files_only not in {"0", "false", "False"}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        add_bos_tokens=False,
        add_eos_tokens=False,
        padding_side=padding_side,
        truncation_side=truncation_side,
        trust_remote_code=True,
        **tokenizer_kwargs,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template_path = f"chat_templates/{model_name_or_path}.jinja"
    if not os.path.exists(template_path):
        logger.warning(
            f"Chat template not found at {template_path}. Using default template."
        )
        return tokenizer

    logger.info(f"Using chat template from {template_path}")
    chat_template = open(template_path).read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = chat_template
    return tokenizer


def get_model(
    model_name_or_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    use_q_lora=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    model_init_kwargs = dict(
        pretrained_model_name_or_path=model_name_or_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=None,
    )
    is_vision_model = check_is_vision_model(model_name_or_path)
    is_gemma4_model = "gemma-4" in model_name_or_path.lower()
    explicit_attention_implementation = (
        model_kwargs is not None and "attn_implementation" in model_kwargs
    )
    if model_kwargs is not None:
        model_init_kwargs.update(model_kwargs)
    local_files_only = os.getenv("HF_LOCAL_FILES_ONLY") or os.getenv("TRANSFORMERS_OFFLINE")
    if local_files_only and "local_files_only" not in model_init_kwargs:
        model_init_kwargs["local_files_only"] = local_files_only not in {"0", "false", "False"}

    is_bidir_model = (
        "bert" in model_name_or_path.lower() or "gte" in model_name_or_path.lower()
    )

    if explicit_attention_implementation:
        pass
    elif use_flash_attn and not is_gemma4_model:
        if "gte" not in model_name_or_path:
            model_init_kwargs["attn_implementation"] = "flash_attention_2"
        elif "gte" in model_name_or_path:
            model_init_kwargs["attn_implementation"] = "sdpa"
    elif is_gemma4_model:
        # Gemma 4 uses Transformers' exact PyTorch FlexAttention backend when
        # the generic acceleration switch is enabled. The pinned environment
        # intentionally does not depend on the separately packaged FA2 wheel.
        model_init_kwargs["attn_implementation"] = (
            "flex_attention" if use_flash_attn else "sdpa"
        )

    if is_vision_model or is_gemma4_model:
        # always use sdpa for vision models
        # model_init_kwargs["attn_implementation"] = "sdpa"
        model_init_kwargs.pop("use_cache")
    elif is_bidir_model:
        model_init_kwargs["torch_dtype"] = torch.float32
        model_init_kwargs.pop("use_cache")

    if use_q_lora:
        # https://huggingface.co/blog/4bit-transformers-bitsandbytes
        # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
        # see bitsandbytes for the quantization implementation https://github.com/bitsandbytes-foundation/bitsandbytes
        # see unsloth https://huggingface.co/docs/trl/v0.7.11/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth
        # does work currently bc it modifies the forward pass call of Linear
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_init_kwargs["quantization_config"] = bnb_config

    logger.debug(f"Model init kwargs: {model_init_kwargs}")
    if not is_vision_model:
        if is_bidir_model:
            model = AutoModel.from_pretrained(**model_init_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(**model_init_kwargs)
    else:
        model = Gemma3ForConditionalGeneration.from_pretrained(**model_init_kwargs)
        model = model.language_model
    if peft_config is not None:
        model = PeftModel(model, peft_config)
    model.train(train)
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad
    return model


def get_lora_config(model_dir, **kwargs):
    if "target_modules" not in kwargs or kwargs["target_modules"] is None:
        logger.info("No target modules specified for LoRA.")
        return None
    r = kwargs.pop("lora_r", 8)
    lora_alpha = kwargs.pop("lora_alpha", None)
    module_name_regex = kwargs.pop("module_name_regex", None)
    target_module_shapes = kwargs.pop("target_module_shapes", None)
    bare_target_modules = kwargs.get("target_modules")
    if module_name_regex and bare_target_modules:
        targets_re = "|".join(bare_target_modules)
        kwargs["target_modules"] = rf".*{module_name_regex.lstrip('^')}.*\.({targets_re})$"
    peft_conf_kwargs = dict(
        r=r,
        peft_type=PeftType.LORA,
        base_model_name_or_path=model_dir,
        task_type="CAUSAL_LM",
        lora_dropout=kwargs.get("lora_dropout", 0.0),
        lora_alpha=lora_alpha if lora_alpha is not None else r ** (3 / 2) * 2,
    )

    peft_conf_kwargs.update(kwargs)
    peft_config = _get_peft_config(peft_conf_kwargs)
    peft_config.module_name_regex = module_name_regex
    peft_config.bare_target_modules = bare_target_modules
    peft_config.target_module_shapes = target_module_shapes
    return peft_config
