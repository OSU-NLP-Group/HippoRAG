---
name: hipporag-memory
description: How to use the HippoRAG MCP tools (retrieve, rag_qa, index, delete, index_stats) as long-term memory. Use when the user asks to remember, index or "add to memory" documents, notes or files, when they ask questions that may be answered by previously indexed material, or when they ask to forget or remove indexed content.
---

# HippoRAG long-term memory

The `hipporag` MCP server keeps a persistent knowledge-graph index on disk. Indexed passages survive across sessions and projects, so check it before saying you don't know something the user may have indexed earlier.

## Choosing a tool

- **`retrieve`**: the default for answering questions. It returns ranked passages with `source_id` and metadata and makes no LLM call. Read the passages, answer in your own words, and cite `source_id`s. Use `top_k` 5–10 for most questions and up to ~20 for broad or multi-hop ones.
- **`rag_qa`**: only when the user explicitly wants HippoRAG's own answer, for example when comparing against the paper's QA pipeline. It costs an extra server-side LLM call, and you still need to check its answer against the returned passages.
- **`index_stats`**: check that the index is non-empty and see which LLM and embedding model it uses. Call it first when a retrieval fails or comes back empty.
- **`index`**: add documents (see below).
- **`delete`**: remove passages by `source_id` (preferred) or by exact passage text as returned by `retrieve`. Deletion cannot be undone, so confirm with the user before deleting anything they did not name explicitly.

## Indexing

- Every **new** passage costs one LLM extraction call on the user's API key. Passages already in the index are skipped. Before indexing more than ~50 passages, tell the user roughly how many there are and get confirmation.
- Always pass objects, not bare strings: `{"text": ..., "source_id": ..., "metadata": {...}}`. Use a stable `source_id` such as an absolute file path or URL, and give every passage from the same source the same `source_id`, so the whole source can later be deleted or refreshed in one call. Put positional info in `metadata`, e.g. `{"chunk": 3, "title": "..."}`.
- Each document becomes one passage; the server does not chunk. Split long text into self-contained passages of roughly 100–400 words at paragraph or section boundaries. A passage should make sense on its own, so carry a heading or title into it when it helps.
- To **update** a changed source, first `delete` its `source_id`, then `index` the new passages. Otherwise the stale passages remain.
- Indexing runs on a single worker, so a large batch delays later tool calls until it finishes. Prefer batches of a few dozen passages.

## Configuration constraints

- An index is tied to the LLM and embedding model it was built with. The server keeps a separate sub-index per `{llm}_{embedding}` pair, so changing either setting in the plugin options switches to a different, possibly empty, index rather than reusing the old one. If `index_stats` unexpectedly shows 0 passages, tell the user this is the likely cause.
- If a tool reports an authentication or connection error, the plugin's API key or base URL option is probably unset or wrong. Ask the user to fix it in the plugin settings (`/plugin` → hipporag → configure) and restart the session.
- The first launch installs HippoRAG and its dependencies (including PyTorch) through `uvx`, which can take several minutes. If the server is not connected yet, say so rather than guessing.
