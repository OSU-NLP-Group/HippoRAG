from functools import partial

import torch
from peft import PeftConfig
from torch import nn

from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers, lora_forward
from ctx_to_lora.modeling.text_to_lora_impl import (
    embed_texts,
    get_layers,
    get_peft_config,
    load_hypermod,
)
from ctx_to_lora.utils import get_peft_modules


class TextToLoRA(nn.Module):
    def __init__(self, model_name_or_path, prefix_tokens, device):
        assert model_name_or_path == "google/gemma-2-2b-it"
        super().__init__()
        hypermod_dir = "trained_t2l/gemma_2b_t2l"
        peft_config = get_peft_config(
            PeftConfig.from_json_file(f"{hypermod_dir}/adapter_config.json")
        )

        # ours lora forward pass uses alpha directly
        peft_config.lora_alpha = peft_config.lora_alpha / peft_config.r
        self.prefix_tokens = prefix_tokens
        self.device = device
        (
            _,
            self.t2l_model,
            self.base_model,
            self.tokenizer,
            self.emb_model,
            self.emb_tokenizer,
            self.task_desc_format_fn,
            self.pooling_fn,
        ) = load_hypermod(hypermod_dir, device)
        layer_indices = range(len(get_layers(self.base_model)))

        self.layer_indices = torch.tensor(
            layer_indices, dtype=torch.long, device=device
        )
        # patch base model forward pass to use lora
        layers = get_layers(self.base_model)
        lora_forward_fn = lora_forward

        for layer_idx in self.layer_indices:
            for module_info in get_peft_modules(layers[layer_idx], peft_config):
                module = module_info["module"]
                module.forward = partial(
                    lora_forward_fn,
                    self=module,
                    lora_dropout_p=peft_config.lora_dropout,
                    scaling=peft_config.lora_alpha,
                )

    @property
    def generation_config(self):
        return self.base_model.generation_config

    def generate_weights(self, ctx_txt: str):
        # generate loras
        ctx_emb = embed_texts(
            [ctx_txt],
            self.emb_model,
            self.emb_tokenizer,
            self.task_desc_format_fn,
            self.pooling_fn,
            self.device,
        )
        encoder_out = self.t2l_model.task_encoder(ctx_emb)
        encoded_task_emb = encoder_out["encoded_task_emb"].detach()

        lora_A, lora_B = dict(), dict()
        lora_dict = dict()
        for target_module in self.t2l_model.target_modules:
            factorized_delta_w = self.t2l_model.get_delta_weights(
                self.layer_indices,
                target_module,
                encoded_task_emb.expand(self.layer_indices.shape[0], -1),
                factorized=True,
            )
            # lora_A[target_module]: [n_layers, r, d_in]
            # lora_A[target_module]: [n_layers, d_out, r]
            lora_A[target_module], lora_B[target_module] = factorized_delta_w

            # convert to lora format used by lora_forward
            # dict of {module:
            #   {A: [bs, n_layers, r, d_inim],
            #    B: [bs, n_layers, r, d_outim]}}
            lora_dict[target_module] = dict(
                A=lora_A[target_module].unsqueeze(0),
                B=lora_B[target_module].transpose(-1, -2).unsqueeze(0),
            )
        return lora_dict

    def generate(self, *args, **kwargs):
        ctx_ids_full = kwargs["ctx_ids"]
        ctx_txt = self.tokenizer.decode(
            ctx_ids_full[0, len(self.prefix_tokens) :], skip_special_tokens=True
        )
        generated_loras = self.generate_weights(ctx_txt)
        apply_lora_to_layers(
            self.base_model,
            self.layer_indices,
            generated_loras,
            n_qs=torch.tensor([1], device=self.device),
            position_ids=None,
        )
        kwargs.pop("ctx_ids", None)
        kwargs.pop("ctx_attn_mask", None)
        kwargs.pop("n_ctx_chunks", None)
        return self.base_model.generate(*args, **kwargs)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from ctx_to_lora.data.definitions import CTX_AFFIXES
    from ctx_to_lora.data.processing import load_and_process_dataset

    model_name = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds = load_and_process_dataset("pwc", split="train", num_proc=8)
    ctx = ds[0]["context"]
    inp = ds[1]["prompts"][0]
    ctx_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": ctx}], return_tensors="pt", return_dict=True
    )
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": ctx + "\n\n" + inp}],
        return_tensors="pt",
        return_dict=True,
    )
    ctx_ids = {k: v.to("cuda") for k, v in ctx_ids.items()}
    input_ids = {k: v.to("cuda") for k, v in input_ids.items()}

    prefix_tokens = CTX_AFFIXES[model_name]["prefix"]
    prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long)

    t2l_model = TextToLoRA(
        model_name,
        prefix_tokens,
        device="cuda",
    )

    with torch.no_grad():
        for _ in range(1):
            outputs = t2l_model.generate(
                **input_ids,
                ctx_ids=ctx_ids["input_ids"],
                max_new_tokens=256,
                do_sample=False,
            )
            print(
                f"Student response: {tokenizer.batch_decode(outputs, skip_special_tokens=False)}"
            )
