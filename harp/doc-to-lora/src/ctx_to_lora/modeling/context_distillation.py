import gc
import json
import logging
import os
import re
from math import ceil
from typing import Any

import torch
from jaxtyping import Integer
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ctx_to_lora.data.definitions import CTX_AFFIXES
from ctx_to_lora.data.q_generation_template import (
    Q_GEN_PROMPT_TEMPLATE,
    Q_GEN_PROMPT_TEMPLATE_REPEAT,
    Q_GEN_SYSTEM_TEMPLATE,
    STOP_STRINGS,
)
from ctx_to_lora.data.self_gen_template import SELF_QA_INTX
from ctx_to_lora.utils import log_num_train_params

logger = logging.getLogger()


def get_q_gen_prompt(context, n_qa_pairs):
    prompt = Q_GEN_PROMPT_TEMPLATE.format(context=context, n_qa_pairs=n_qa_pairs)
    return prompt


def get_q_gen_prompt_repeat(context, qa_pairs, n_qa_pairs):
    example_qa_pairs = ""
    for i, (q, a) in enumerate(qa_pairs, 1):
        example_qa_pairs += f"Question {i}: {q}\nAnswer {i}: {a}\n"
    prompt = Q_GEN_PROMPT_TEMPLATE_REPEAT.format(
        context=context,
        qa_pairs=example_qa_pairs,
        n_qa_pairs=n_qa_pairs,
    )
    return prompt


def check_should_skip(txt: str, vllm_model: str) -> bool:
    """Check if the response should be skipped based on stop strings."""
    for stop in STOP_STRINGS[vllm_model]:
        if stop in txt[-len(stop) :]:
            return (txt.split(stop)[0], False)  # Found a valid stop string
    return (txt, True)  # No valid stop string found, skip this response


def postprocess_qa_pairs(res_txt: str):
    """
    Postprocesses the QA pairs from the response text.

    Args:
        res_txt: The response text.
        n_qa_pairs: The number of QA pairs.

    Returns:
        A tuple of two lists, the first containing the questions and the second containing the answers.
    """
    # capture everything after each "Question {number}:" until "Answer"
    q_pattern = r"Question \d+:(.*?)(?=Answer|$)"  # thanks chatgpt
    questions = re.findall(q_pattern, res_txt, flags=re.S)

    a_pattern = r"Answer \d+:(.*?)(?=Question|$)"  # thanks chatgpt
    answers = re.findall(a_pattern, res_txt, flags=re.S)

    if len(questions) != len(answers):
        print(f"Warning---number of questions and answers do not match")
        print(f"Number of questions: {len(questions)}")
        print(f"Number of answers: {len(answers)}")

    out_q = []
    out_a = []
    n_skips = 0
    if (len(questions) > 0) and (len(answers) > 0):
        n_gen_pairs = min(len(questions), len(answers))
        has_left_over = n_gen_pairs < len(questions) or n_gen_pairs < len(answers)
        for i in range(n_gen_pairs):
            response = answers[i].strip()
            question = questions[i].strip()
            if not response or not question:
                print(f"Skipping empty question or answer at index {i}")
                continue
            if (not has_left_over) and (i == n_gen_pairs - 1):
                response, skip = check_should_skip(response, "google/gemma-3-12b-it")
                if skip:
                    print(f"Skipping due to missing stop string")
                    n_skips += 1
                    continue
            out_q.append(question.strip())
            out_a.append(response.strip())
    print(f"Skipped {n_skips} responses due to missing stop strings")

    return out_q, out_a


def build_messages(ctx_text: str, level: int, example_qa_pairs: list = None):
    messages = [
        {"role": "system", "content": Q_GEN_SYSTEM_TEMPLATE},
        {
            "role": "user",
            "content": get_q_gen_prompt(ctx_text, 5)
            if level == 0
            else get_q_gen_prompt_repeat(ctx_text, example_qa_pairs, 5),
        },
    ]
    return messages


def get_shifted_label_pos(labels):
    pos = torch.where(labels != -100)
    # (batch_idx, token_idx)
    return (pos[0], pos[1] - 1)


def logits_at_positions(outputs: ModelOutput, pos) -> Tensor:
    logits = outputs.logits
    return logits[pos[0], pos[1]]


def ctx_inp_split(
    ctx_inp_ids, ctx_inp_sep_seq, pad_token_id, prefix_tokens=None, padding_side="right"
):
    # Split each row in ctx_inp_ids at the first occurrence of ctx_inp_sep_seq
    # Return the part after the separator for each row
    batch_size = ctx_inp_ids.size(0)
    sep_len = ctx_inp_sep_seq.size(0)
    out_inp = []
    out_ctx = []
    for i in range(batch_size):
        row = ctx_inp_ids[i]
        # Find where the separator starts
        for j in range(row.size(0) - sep_len + 1):
            if torch.equal(row[j : j + sep_len], ctx_inp_sep_seq):
                out_ctx.append(row[:j])
                if prefix_tokens is not None:
                    out_inp.append(
                        torch.cat([prefix_tokens, row[j + sep_len :]], axis=-1)
                    )
                else:
                    out_inp.append(row[j + sep_len :])
                break
        else:
            # If separator not found
            raise ValueError(f"Separator sequence not found in row {i}")
    out_inp = torch.nn.utils.rnn.pad_sequence(
        out_inp, batch_first=True, padding_value=pad_token_id, padding_side=padding_side
    )
    out_ctx = torch.nn.utils.rnn.pad_sequence(
        out_ctx, batch_first=True, padding_value=pad_token_id, padding_side=padding_side
    )
    return out_ctx, out_inp


def get_peft_layers(model, peft_config):
    out = []
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules
        out.append(module)

    return out


class CtxDistillModel(nn.Module):
    def __init__(
        self,
        base_model: PeftModel,
        prefix_tokens: Integer[Tensor, "n"],
        ctx_inp_sep_seq: Integer[Tensor, "m"],
        pad_token_id: int,
        update_iterations: int,
        reset: bool = True,
        tokenizer=None,
        q_model: PreTrainedModel | None = None,
        q_tokenizer=None,
        reprompt_ctx: bool = False,
        lora_save_dir: str | None = None,
        save_after_distill: bool = True,
        q_gen_rounds: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        self.register_module("base_model", base_model)
        self.register_module("q_model", q_model)
        self.register_buffer("prefix_tokens", prefix_tokens)
        self.register_buffer("ctx_inp_sep_seq", ctx_inp_sep_seq)
        self.tokenizer = tokenizer
        self.q_tokenizer = q_tokenizer
        self.pad_token_id = pad_token_id
        self.update_iterations = update_iterations
        self.reprompt_ctx = reprompt_ctx
        self.reset = reset
        self.device = base_model.device
        self.to(self.device)
        self.q_gen_rounds = q_gen_rounds
        # New save options
        self.lora_save_dir = lora_save_dir
        self.save_after_distill = save_after_distill

        self.peft_config = base_model.peft_config["default"]
        self.adapter_name = "default"
        self.base_model.set_adapter("default")
        for layer in get_peft_layers(self.base_model, self.peft_config):
            for name, p in layer.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    p.requires_grad = True
        log_num_train_params(self.base_model)
        self._init_optim()
        # Mini-batch size for distillation updates
        self.batch_size = batch_size

    @property
    def generation_config(self):
        return self.base_model.generation_config

    def _init_optim(self):
        self.optimizer = torch.optim.AdamW(
            [
                p
                for l in get_peft_layers(self.base_model, self.peft_config)
                for p in l.parameters()
                if p.requires_grad
            ],
            lr=1e-4,
        )

    def reset_lora(self):
        print("Resetting LoRA")
        for layer in get_peft_layers(self.base_model, self.peft_config):
            layer.reset_lora_parameters(self.adapter_name, init_lora_weights=True)
        self._init_optim()

    def save_lora(self):
        """
        Save current LoRA adapter in PEFT format plus a lightweight JSON summary
        for easy human inspection/manipulation.
        """
        if self.lora_save_dir is None:
            return
        os.makedirs(self.lora_save_dir, exist_ok=True)
        # Standard PEFT save (produces adapter_config.json + adapter_model.bin / safetensors)
        self.base_model.save_pretrained(self.lora_save_dir)
        # Human-readable summary of LoRA parameter shapes
        summary = {
            name: list(p.shape)
            for name, p in self.base_model.named_parameters()
            if ("lora_A" in name or "lora_B" in name) and p.requires_grad
        }
        with open(os.path.join(self.lora_save_dir, "lora_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"LoRA adapter saved to {self.lora_save_dir}")

    @torch.enable_grad()
    def _distill_context(
        self,
        ctx_inp_res_ids: Integer[Tensor, "bs ctx_inp_length"],
        ctx_inp_res_attention_mask: Integer[Tensor, "bs ctx_inp_length"],
        teacher_labels: Integer[Tensor, "bs ctx_inp_length"],
        inp_res_ids: Integer[Tensor, "bs inp_length"],
        inp_res_attention_mask: Integer[Tensor, "bs inp_length"],
        student_labels: Integer[Tensor, "bs inp_length"],
    ):
        # Implements KD-style loss by computing teacher (with context) and student (no context)
        # log-probs locally, using mini-batches for updates.

        was_training = self.training
        self.train()

        num_samples = ctx_inp_res_ids.size(0)
        mb = self.batch_size
        num_batches = ceil(num_samples / mb)

        total_steps = max(self.update_iterations * num_batches, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=0.0
        )

        print(
            f"Starting context distillation for {self.update_iterations} epochs, "
            f"mini-batch size {mb} ({num_batches} batches/epoch), "
            f"cosine LR schedule over {total_steps} steps"
        )

        for epoch in range(self.update_iterations):
            # Shuffle order each epoch for SGD
            perm = torch.randperm(num_samples, device=self.device)

            epoch_loss = 0.0
            for b in range(num_batches):
                start = b * mb
                end = min(start + mb, num_samples)
                indices = perm[start:end]

                b_ctx_ids = ctx_inp_res_ids[indices]
                b_ctx_am = ctx_inp_res_attention_mask[indices]
                b_teacher_labels = teacher_labels[indices]

                b_inp_ids = inp_res_ids[indices]
                b_inp_am = inp_res_attention_mask[indices]
                b_student_labels = student_labels[indices]

                # Compute teacher distribution (top-k) for this mini-batch
                with torch.no_grad(), self.base_model.disable_adapter():
                    t_pos = get_shifted_label_pos(b_teacher_labels)
                    teacher_outputs = self.base_model(
                        b_ctx_ids, attention_mask=b_ctx_am
                    )
                    teacher_logits = logits_at_positions(teacher_outputs, t_pos)
                    K = 16
                    topk_vals, topk_idx = teacher_logits.topk(K, dim=-1)
                    teacher_denom = torch.logsumexp(
                        teacher_logits.float(), dim=-1, keepdim=True
                    )
                    teacher_p = (topk_vals - teacher_denom).exp().detach()  # [N, K]

                # Student forward and update for this mini-batch
                self.optimizer.zero_grad()
                s_pos = get_shifted_label_pos(b_student_labels)
                student_outputs = self.base_model(b_inp_ids, attention_mask=b_inp_am)
                student_logits = logits_at_positions(student_outputs, s_pos)
                student_denom = torch.logsumexp(
                    student_logits.float(), dim=-1, keepdim=True
                )
                selected_student_logits = student_logits.gather(-1, topk_idx)
                student_logq = selected_student_logits - student_denom  # [N, K]
                token_losses = -(teacher_p * student_logq).sum(dim=-1)  # [N]
                loss = token_losses.mean()

                loss.backward()
                self.optimizer.step()
                scheduler.step()

                epoch_loss += loss.detach().item()

            cur_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{self.update_iterations}, "
                f"batch {b + 1}/{num_batches}: loss={epoch_loss / num_batches:.4f}, lr={cur_lr:.6e}"
            )

        if not was_training:
            self.eval()

    def generate_questions(self, *args, **kwargs):
        questions = self.q_model.generate(*args, **kwargs)
        return questions

    def teacher_generate(self, *args, **kwargs):
        # rename for separate timing
        return self.base_model.generate(*args, **kwargs)

    def student_generate(self, *args, **kwargs):
        # rename for separate timing
        return self.base_model.generate(*args, **kwargs)

    def get_lora_state(self, clone: bool = True):
        """
        Return a dict of current LoRA parameter tensors.
        clone=True returns detached cloned tensors (safe to store).
        """
        return {
            name: (p.detach().clone() if clone else p)
            for name, p in self.base_model.named_parameters()
            if ("lora_A" in name or "lora_B" in name)
        }

    def generate(
        self,
        *model_inputs_args: Any,
        distill_only: bool = False,
        **model_inputs_kwargs: dict[str, Any],
    ):
        if self.reset:
            self.reset_lora()

        # teacher tokens
        orig_ctx_inp_ids = model_inputs_kwargs.pop("input_ids")
        ctx_inp_ids = orig_ctx_inp_ids.clone()
        _, orig_inp_ids = ctx_inp_split(
            ctx_inp_ids,
            self.ctx_inp_sep_seq,
            self.pad_token_id,
            self.prefix_tokens,
            padding_side="left",
        )
        ctx_inp_attention_mask = model_inputs_kwargs.pop("attention_mask")

        if self.q_model is not None:
            self.q_model.to(self.base_model.device)
            # Extract context-only portion after separator (remove prefix tokens from first row)
            ctx_ids_full, _ = ctx_inp_split(
                ctx_inp_ids, self.ctx_inp_sep_seq, self.pad_token_id
            )  # [bs, var_len]
            ctx_ids = ctx_ids_full[0, len(self.prefix_tokens) :]
            ctx_txt = self.tokenizer.decode(ctx_ids, skip_special_tokens=True)
            questions = []
            answers = []
            # Build multiple instruction variants
            for lvl in range(self.q_gen_rounds):
                messages = build_messages(
                    ctx_txt, lvl, zip(questions, answers) if lvl > 0 else None
                )
                q_inputs = self.q_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                q_inputs = {k: v.to(self.q_model.device) for k, v in q_inputs.items()}

                with torch.no_grad():
                    question_outputs = self.generate_questions(
                        input_ids=q_inputs["input_ids"],
                        attention_mask=q_inputs["attention_mask"],
                        max_new_tokens=1024,
                        do_sample=False,
                        temperature=0.0,
                        eos_token_id=106,  # <end_of_turn> for gemma-3-12b-it
                    )
                # Slice off the prompt portion
                gen_only = question_outputs[:, q_inputs["input_ids"].shape[-1] :]
                res = self.q_tokenizer.batch_decode(gen_only, skip_special_tokens=False)
                gen_q_list, gen_a_list = postprocess_qa_pairs(res[0])
                questions += gen_q_list
                answers += gen_a_list

            if len(questions) == 0:
                # when q_model refuses to provide questions
                # only happens with sample 116 in longbench/multifieldqa_en_e
                # in this case cd just doesn't work, we fall back to zero-shot answer
                print(f"Warning---no questions generated, skipping distillation")
                attention_mask = torch.where(
                    orig_inp_ids != self.pad_token_id, 1, 0
                ).long()
                return self.student_generate(
                    orig_inp_ids, attention_mask=attention_mask, **model_inputs_kwargs
                )

            ctx_inp_messages = [
                [{"role": "user", "content": f"{ctx_txt}\n\n{SELF_QA_INTX}\n\n{q}"}]
                for q in questions
            ]
            encoded_ctx_inp = self.tokenizer.apply_chat_template(
                ctx_inp_messages,
                tokenize=True,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            encoded_ctx_inp = {k: v.to(self.device) for k, v in encoded_ctx_inp.items()}
            ctx_inp_ids = encoded_ctx_inp["input_ids"]
            ctx_inp_attention_mask = encoded_ctx_inp["attention_mask"]
            self.q_model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        # sample responses first
        ctx_inp_res_ids = self.teacher_generate(
            ctx_inp_ids,
            attention_mask=ctx_inp_attention_mask,
            **model_inputs_kwargs,
        )
        ctx_inp_res_attention_mask = torch.where(
            ctx_inp_res_ids != self.pad_token_id, 1, 0
        ).long()
        ctx_inp_res_txt = self.tokenizer.batch_decode(ctx_inp_res_ids)

        bs = ctx_inp_ids.shape[0]
        res_len = ctx_inp_res_ids.shape[-1] - ctx_inp_ids.shape[-1]
        res_ids = ctx_inp_res_ids[:, -res_len:]  # correct

        pads = torch.full_like(ctx_inp_ids, self.pad_token_id)
        teacher_labels = torch.cat([pads, res_ids], dim=-1)
        teacher_labels = torch.where(
            teacher_labels != self.pad_token_id, teacher_labels, -100
        )

        # student tokens
        _, inp_res_ids = ctx_inp_split(
            ctx_inp_res_ids,
            self.ctx_inp_sep_seq,
            self.pad_token_id,
            self.prefix_tokens,
            padding_side="left",
        )
        inp_res_attention_mask = torch.where(
            inp_res_ids != self.pad_token_id, 1, 0
        ).long()

        student_labels = inp_res_ids.clone()
        student_labels[:, :-res_len] = -100
        student_labels = torch.where(
            student_labels != self.pad_token_id, student_labels, -100
        )

        self._distill_context(
            ctx_inp_res_ids,
            ctx_inp_res_attention_mask,
            teacher_labels,
            inp_res_ids,
            inp_res_attention_mask,
            student_labels,
        )
        # Save LoRA after distillation if requested
        if distill_only:
            return self.get_lora_state()

        model_inputs_kwargs.pop("attention_mask", None)
        model_inputs_kwargs.pop("input_ids", None)
        if self.reprompt_ctx:
            attention_mask = torch.where(orig_ctx_inp_ids != self.pad_token_id, 1, 0)
            model_outputs = self.student_generate(
                orig_ctx_inp_ids, attention_mask=attention_mask, **model_inputs_kwargs
            )
        else:
            attention_mask = torch.where(orig_inp_ids != self.pad_token_id, 1, 0).long()
            model_outputs = self.student_generate(
                orig_inp_ids, attention_mask=attention_mask, **model_inputs_kwargs
            )
        return model_outputs


if __name__ == "__main__":
    from ctx_to_lora.data.processing import load_and_process_dataset
    from ctx_to_lora.model_loading import get_lora_config, get_model_and_tokenizer

    model_name = "google/gemma-2-2b-it"
    q_model_name = "google/gemma-3-12b-it"
    peft_config = get_lora_config(
        model_name, r=8, target_modules=["down_proj"], lora_dropout=0.0
    )
    peft_config.lora_alpha = 16
    model, tokenizer = get_model_and_tokenizer(
        model_name,
        train=False,
        requires_grad=False,
        peft_config=peft_config,
    )
    q_model, q_tokenizer = get_model_and_tokenizer(
        q_model_name,
        train=False,
        requires_grad=False,
        peft_config=peft_config,
    )

    ds = load_and_process_dataset("pwc", split="train", num_proc=8)
    ctx = ds[0]["context"]
    inp = ds[1]["prompts"][0]
    # Build a simple context/input pair separated by a unique token sequence
    sep_text = SELF_QA_INTX
    # ctx = "# Provided Information\nMy name is Tan."
    # ctx = "# Provided Info"
    prompt = f"{ctx}\n\n{sep_text}\n\n{inp}"
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    # encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    prefix_tokens = CTX_AFFIXES[model_name]["prefix"]
    prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long)

    sep_ids = (
        tokenizer(sep_text.strip("\n"), add_special_tokens=False, return_tensors="pt")
        .input_ids[0]
        .to(model.device)
    )

    cd_model = CtxDistillModel(
        base_model=model,
        prefix_tokens=prefix_tokens,
        ctx_inp_sep_seq=sep_ids,
        pad_token_id=tokenizer.pad_token_id,
        update_iterations=100,
        q_model=q_model,
        q_tokenizer=q_tokenizer,
        tokenizer=tokenizer,
        reprompt_ctx=False,
        lora_save_dir="./saved_lora_adapter",  # example save path
        save_after_distill=True,
    )

    with torch.no_grad():
        for _ in range(1):
            base_model_res = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=256,
                do_sample=False,
            )
            print(
                f"Base model response:{tokenizer.batch_decode(base_model_res, skip_special_tokens=False)}"
            )

            outputs = cd_model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=256,
                do_sample=False,
            )
            print(
                f"Student response: {tokenizer.batch_decode(outputs, skip_special_tokens=False)}"
            )
