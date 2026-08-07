# RepoQA 104 evaluation panel

`swefixer_targeted_retrieval_104_step500.jsonl` is the frozen panel used for
the principal HARP evaluation. It contains four repository-grounded facts for
each of 26 held-out SWE-fixer repositories.

Each row includes the open question and gold answer, evidence paths and spans,
answer type, capability family, and frozen MCQ choices. The panel contains:

- 26 behavior and test-as-specification questions;
- 46 cross-file relationship questions;
- 32 import, export, and entrypoint questions.

The file was regenerated using the checked-in deterministic selector and the
original frozen `memory_discovery.jsonl` source. Its SHA-256 is:

```text
d73739eaa2fed11b4d9c700776c10784cf82be63f6fed095a164b1dcc4f0e6c3
```

This matches the recorded hash of the panel used by the evaluations.

