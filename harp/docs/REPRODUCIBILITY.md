# Reproducibility contract

The cleaned source preserves the experiment implementation while keeping
runtime artifacts external.

## Frozen model identity

The Doc-to-LoRA and Code2LoRA configurations use:

```text
model: google/gemma-4-E2B-it
revision: 3e22461f65e89153144f8adb70e3b8c2cc9845a7
```

## Required immutable artifacts

A reproducible run records the hashes of:

- the corpus `READY.json` manifest;
- all referenced QA shards and canonical chunk indexes;
- the Code2LoRA repository-embedding table and its manifest;
- the selected method configuration;
- the HARP source commit;
- the starting checkpoint when a stage is initialized or resumed.

The `READY.json` contract freezes repository identity, snapshot identity,
logical QA IDs, source-family weights, response boundaries, stage assignment,
and deterministic DDP ordering.

## Source provenance

`doc-to-lora/` was derived from upstream Doc-to-LoRA commit `baa85db` and
contains the HARP experiment modifications. `code2lora/` contains the exact
checkpoint-compatible Code2LoRA Python package used by HARP. Both directories
are tracked directly by the single HARP repository.

No generated data, checkpoints, predictions, reports, or numerical findings
belong in either source repository.
