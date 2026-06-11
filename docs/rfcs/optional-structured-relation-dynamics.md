# RFC: Optional Structured Relation Dynamics For HippoRAG Retrieval

Status: upstream RFC draft

Tracking issue: https://github.com/OSU-NLP-Group/HippoRAG/issues/181

## 1. Context & Motivation

HippoRAG already combines several retrieval signals: dense passage retrieval, OpenIE facts, entity graph links, reranking, Personalized PageRank, and QA.

That makes HippoRAG a strong candidate for an optional relation-dynamics layer. The goal is not to replace retrieval. The goal is to expose how retrieval signals support, conflict, weaken, or require downstream verification.

The proposed upstream direction is a small, disabled-by-default hook in HippoRAG retrieval/ranking that can produce structured retrieval metadata and, later, optionally influence ranking when explicitly enabled.

## 2. Problem Statement

A final retrieval score does not show enough of the reasoning surface when:

- dense retrieval and graph/PPR prefer different passages;
- facts are sparse or missing;
- entity links are weak;
- reranking keeps a result without explaining conflict;
- downstream verification fails after an apparently strong retrieval result;
- a user or student needs to understand why a proof path helped or failed.

HippoRAG can return useful context, but downstream systems need a more structured signal trace to decide whether to answer, verify, retrieve more, or explain uncertainty.

## 3. Proposed Solution

Add an optional retrieval relation dynamics hook to HippoRAG.

The hook should be able to expose:

- structured retrieval metadata;
- retrieval signal trace;
- support, indeterminacy, contradiction, channel conflict;
- optional downstream verification friction supplied by an external tool;
- optional relation weighting around the final retrieval/ranking path.

The default behavior must remain unchanged. If disabled, HippoRAG should behave exactly as it does today.

Public language should stay generic: `relation_dynamics`, `retrieval_signal_trace`, `structured_retrieval_metadata`. Theory-specific names should remain in documentation and external research adapters.

## 4. Academic Foundation

The internal mathematical foundation comes from Smarandache's work on Plithogeny, n-Power Set, n-SuperHyperGraph, and Plithogenic n-SuperHyperGraph.

Key sources:

- Plithogeny PDF: https://fs.unm.edu/Plithogeny.pdf
- Plithogeny arXiv: https://arxiv.org/abs/1808.03948
- n-SuperHyperGraph PDF: https://fs.unm.edu/NSS/n-SuperHyperGraph.pdf
- n-SuperHyperGraph / Plithogenic n-SuperHyperGraph UNM entry: https://digitalrepository.unm.edu/math_fsp/348/
- n-SuperHyperGraph PhilArchive entry: https://philarchive.org/rec/SMAITT-3

The local QuaNThoR layer may model a relation state as:

`q=(T,I_system_S,D_f,dF)`

Where:

- `T` is observed support;
- `I_system_S` is system-bound indeterminacy;
- `D_f` is local deformation;
- `dF` is verifier/downstream friction;
- `F` remains external damping/opposition;
- `i_fractal` remains derived.

The hierarchy must not collapse:

`I -> I_system^S -> D_f -> dF -> i_fractal`

## 5. HippoRAG Mapping

| HippoRAG signal | Relation-dynamics meaning |
|---|---|
| dense passage score | semantic support |
| fact score | extracted fact support |
| entity graph links | associative support |
| PPR score | graph-central support |
| rerank behavior | recognition/filtering signal |
| missing facts/entities | indeterminacy |
| dense/PPR disagreement | channel conflict |
| downstream verifier failure | verification friction |

The proposed hook should not require HippoRAG to know Mizar, QuaNThoR, or neutrosophic terminology. It only needs to expose enough structured signal data for optional downstream or experimental relation dynamics.

## 6. Why HippoRAG

HippoRAG is a better fit than a plain vector-only RAG system because its retrieval already has graph, fact, entity, dense, and PPR components.

Those components naturally create relationships among evidence. A relation-dynamics hook would make those relationships inspectable and eventually measurable.

For education, this matters because a proof-learning system should not only say "this passage is relevant"; it should help show why the passage supports, misleads, or needs verification.

## 7. Existing HippoRAG Concepts That Fit

The likely fit points are additive:

- result objects such as `QuerySolution`;
- passage, fact, and entity scores;
- PPR passage ranking;
- reranking metadata;
- central config surfaces;
- graph-search and retrieval phases before final answer generation.

The RFC should ask maintainers where such a hook belongs rather than assume the final implementation location.

## 8. Explicit Constraints

This RFC does not propose:

- changing default retrieval behavior;
- replacing HippoRAG ranking;
- requiring QuaNThoR;
- adding mandatory heavy dependencies;
- exposing theory-heavy API names;
- claiming retrieval improvement before benchmark;
- claiming quantum computation;
- opening a PR before local evidence.

The feature must be opt-in, measurable, and removable without breaking existing users.

## 9. Prior Art & Uniqueness

Relevant prior art includes:

- QuatE / Quaternion Knowledge Graph Embeddings: https://arxiv.org/abs/1904.10281
- Quaternion Graph Neural Networks: https://proceedings.mlr.press/v157/nguyen21a.html
- Quantum-inspired information retrieval: https://arxiv.org/abs/1310.3001

This project must not claim those areas as new.

The specific contribution under study is narrower: combine HippoRAG's multi-signal retrieval with plithogenic contradiction thinking, optional relation dynamics, and downstream verification friction in a form that can be measured locally before any upstream request.

## 10. Open Questions For Maintainers

If this is posted upstream, ask:

1. Would HippoRAG maintainers accept an optional retrieval relation dynamics hook?
2. Should this live in result metadata, a callback, a config-gated module, or an external adapter?
3. Which retrieval signals can be exposed without runtime or compatibility cost?
4. Should downstream verification friction be accepted as external metadata only?
5. Should the hook ever influence ranking, or remain trace-only?
6. What naming would be acceptable for a public API?
7. Would maintainers prefer an RFC, example adapter, or experimental branch before code?
8. What tests would prove default behavior remains unchanged?

## 11. Next Step

This RFC is staged as a docs-only upstream discussion artifact.

Next steps:

1. Clean QuaNThoR root and document its public archive intent.
2. Build a small educational Mizar/QuaNThoR seed dataset.
3. Run HippoRAG retrieval against the dataset.
4. Compare baseline retrieval, neutrosophic audit, and plithogenic quaternion trace.
5. Record whether relation dynamics predicts verifier friction better than scalar scores alone.
6. Decide whether evidence justifies posting this RFC upstream.

QuaNThoR is the local public validation lab for this work; it is not an upstream requirement.
