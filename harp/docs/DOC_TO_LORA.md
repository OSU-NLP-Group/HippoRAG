# Doc-to-LoRA

The HARP Doc-to-LoRA fork turns a complete frozen repository snapshot into a
question-independent adapter for a frozen Gemma answer model.

## Adapter generation

Each canonical repository chunk passes through the frozen context encoder. A
trainable Perceiver consumes selected per-layer activations and produces a
rank-8 LoRA for each configured answer-model projection. The production setup
targets the large language-model `down_proj` projection in 20 Gemma layers.

The Perceiver uses eight rank latents of width 512, nine cross-attention blocks,
per-rank generation, and per-layer processing. Context layers are processed
sequentially to bound activation memory; every target depth receives its own
generated factors.

## Repository composition

The fork supports three composition modes over per-chunk dense updates
`delta_W = B @ A`:

- `concat`: concatenate the chunk factors, so repository adapter rank grows
  with the number of chunks;
- `ties`: trim and sign-merge the dense chunk updates, then reconstruct a
  fixed-rank adapter using truncated SVD;
- `streaming_ties_exact`: retain exact sign counts and separate positive and
  negative magnitude sums while chunks stream through the model, choose the
  final sign after all chunks, and perform the same fixed-rank reconstruction.

The learned context-independent rank-8 bias adapter is added once per
repository, not once per chunk.

## Training objective

The retained production configuration uses answer-token cross-entropy. Each
logical QA contributes its mean answer-token loss. Deterministic-family rows
have weight `1.0`; LLM-family rows have weight `1.8`. The Gemma context encoder
and answer model remain frozen; the Perceiver and hypernetwork are trained.

Training is a two-stage curriculum:

- `stage1`: snapshots with at most two chunks;
- `stage2a`: snapshots with three through nine chunks, initialized from the
  completed Stage 1 model weights.

## Configurations

```text
doc-to-lora/configs/repoqa/gemma4_e2b_snapshot_memory_64k_k9_10m.yaml
doc-to-lora/configs/repoqa/gemma4_e2b_snapshot_memory_64k_k9_10m_ties_r64.yaml
doc-to-lora/configs/repoqa/gemma4_e2b_snapshot_memory_64k_k9_10m_streaming_ties_exact_r64.yaml
```

Dataset and output paths in a config are overridden by the generic Slurm
launcher. See `LAUNCHING.md`.

