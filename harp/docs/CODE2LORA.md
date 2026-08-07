# Code2LoRA

The retained Code2LoRA implementation maps a fixed repository embedding
directly to LoRA factors for a frozen Gemma answer model.

## Architecture

- repository representation: 2,048-dimensional validated embedding;
- generator: a two-layer, 1,024-dimensional hypernetwork;
- generated adapter: rank 8 with alpha 16;
- targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, and
  `down_proj` across the Gemma language model;
- frozen components: Gemma backbone, embeddings, and output head.

The checkpoint-compatible Python package remains named `repotune_issuefix`.
Changing that import namespace is deliberately avoided so existing checkpoints
and serialized metadata continue to load.

## Training

`repotune_issuefix.train_code2lora_repoqa` trains on the same frozen QA corpus
as Doc-to-LoRA. It generates the repository adapter once for a length-bucketed
QA pack and applies weighted answer-token cross-entropy.

The retained training defaults are AdamW, learning rate `1e-4`, weight decay
`0.01`, three-percent warmup, cosine decay, gradient clipping at `1.0`, and a
ten-million-QA exposure target. Checkpoints are exposure-addressed so a run can
resume without changing corpus order.

## Source layout

```text
code2lora/src/repotune_issuefix/code2lora_gemma.py
code2lora/src/repotune_issuefix/train_code2lora_repoqa.py
code2lora/src/repotune_issuefix/repoqa_baselines.py
code2lora/src/repotune_issuefix/build_repo_embeddings.py
```

Use `scripts/train_code2lora.sbatch` for the experiment configuration.

