import torch
from llmlingua import PromptCompressor
from torch import nn

from ctx_to_lora.data.definitions import CTX_AFFIXES


class LLMLinguaModel(nn.Module):
    def __init__(self, model, tokenizer, compression_rate):
        super().__init__()
        self.base_model = model
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,  # Whether to use llmlingua-2
        )
        model_name = self.base_model.name_or_path
        self.register_buffer("prefix", torch.tensor(CTX_AFFIXES[model_name]["prefix"]))
        self.register_buffer("suffix", torch.tensor(CTX_AFFIXES[model_name]["suffix"]))
        self.len_prefix = len(self.prefix)
        self.len_suffix = len(self.suffix)
        self.tokenizer = tokenizer
        self.compression_rate = compression_rate

    @property
    def generation_config(self):
        return self.base_model.generation_config

    def compress(self, prompt_txt: str, rate: float):
        return self.compressor.compress_prompt(
            prompt_txt, rate=rate, force_tokens=["\n", "?"]
        )

    def compress_tokens(self, input_ids, query_text):
        bs = input_ids.shape[0]
        txt = self.tokenizer.batch_decode(
            input_ids[:, self.len_prefix : -self.len_suffix]
        )
        q_start_idx = txt[0].rfind(query_text)
        ctx_txt = txt[0][:q_start_idx]
        q_txt = txt[0][q_start_idx:]
        compressed_txt = self.compress(ctx_txt, rate=self.compression_rate)
        compressed_ids = self.tokenizer(
            compressed_txt["compressed_prompt"] + "\n\n" + q_txt,
            return_attention_mask=False,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.base_model.device)

        out = torch.cat(
            [self.prefix.expand(bs, -1), compressed_ids, self.suffix.expand(bs, -1)],
            dim=-1,
        )
        return out

    def generate(self, *args, **kwargs):
        # take ctx_ids
        # strip prefix and suffix
        # ctx_ids is left padded
        ctx_ids = kwargs["ctx_ids"][:, self.len_prefix : -self.len_suffix]
        # decode ctx_ids to ctx_txt
        ctx_txt = self.tokenizer.batch_decode(ctx_ids)
        compressed_ctx_txt = self.compress(ctx_txt, rate=self.compression_rate)
        compressed_ctx_ids = self.tokenizer(
            compressed_ctx_txt["compressed_prompt"] + "\n\n",
            return_attention_mask=False,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.base_model.device)

        bs = ctx_ids.shape[0]
        ctx_inp_ids = torch.cat(
            [
                self.prefix.expand(bs, -1),
                compressed_ctx_ids["input_ids"],
                kwargs["input_ids"][:, self.len_prefix :],
            ],
            dim=-1,
        )
        attn_mask = torch.ones_like(ctx_inp_ids)
        for k in [
            "ctx_ids",
            "ctx_attn_mask",
            "n_ctx_chunks",
            "input_ids",
            "attention_mask",
        ]:
            kwargs.pop(k, None)
        return self.base_model.generate(ctx_inp_ids, attention_mask=attn_mask, **kwargs)


if __name__ == "__main__":
    from ctx_to_lora.model_loading import get_model_and_tokenizer

    model, tokenizer = get_model_and_tokenizer(
        "google/gemma-2-2b-it",
        train=False,
        requires_grad=False,
    )

    # Demo: build wrapper, create a toy context + prompt, run compression + generation.
    device = "cuda"
    llm = LLMLinguaModel(model, tokenizer).to(device)

    # Toy context and user prompt
    context_text = (
        "This is a short illustrative context about large language models and compression. "
        "They can reduce prompt length while preserving meaning."
    )
    user_prompt = (
        "Summarize the context in one concise sentence."  # what we want model to do
    )

    # Tokenize raw context (core) without special tokens
    core_ctx_ids = tokenizer.apply_chat_template(
        [[{"role": "user", "content": context_text}]],
        tokenize=True,
        add_generation_prompt=True,
        return_attention_mask=False,
        padding=False,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    # Input prompt tokens (what follows the contextual block)
    input_ids = tokenizer.apply_chat_template(
        [[{"role": "user", "content": user_prompt}]],
        tokenize=True,
        add_generation_prompt=True,
        return_attention_mask=False,
        padding=False,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    print("Original context length (chars):", len(context_text))

    # Run generation (may vary depending on model capabilities)
    output_ids = llm.generate(ctx_ids=core_ctx_ids, input_ids=input_ids)
    # Decode only the tail beyond supplied input for readability
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"\nFull generated text:\n{generated_text}")
