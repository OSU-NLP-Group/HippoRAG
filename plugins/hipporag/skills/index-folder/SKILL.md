---
name: index-folder
description: Index the text files in a folder into HippoRAG long-term memory.
argument-hint: "<folder> [glob, e.g. **/*.md]"
disable-model-invocation: true
---

# Index a folder into HippoRAG

Arguments: `$ARGUMENTS`. The first argument is the folder; the optional second one is a glob pattern. If no glob is given, use `**/*.{md,markdown,txt,rst}`. If no folder is given, ask for one.

1. **Collect files.** List the files matching the glob under the folder. Skip hidden directories, `node_modules`, `.venv`, build output, binary files and files over ~1 MB. Resolve each path to an absolute path.
2. **Split into passages.** Read each file and split it at headings and paragraph boundaries into self-contained passages of roughly 100–400 words. Prefix each passage with the nearest heading when the passage would otherwise lack context. Drop passages that are empty or pure boilerplate.
3. **Confirm the cost.** Call `index_stats` to see the current size and model. Then tell the user how many files and passages you are about to index, and that each new passage costs one LLM extraction call with the configured model (passages already indexed are skipped). Wait for confirmation if there are more than ~50 passages.
4. **Refresh changed files.** Ask whether files already in the index should be refreshed. If so, `delete` them by `source_id` first, so edited files don't leave stale passages behind. Skip this step on a first-time index.
5. **Index.** Call `index` in batches of about 20–30 passages. Each document is `{"text": <passage>, "source_id": <absolute file path>, "metadata": {"chunk": <index within file>, "path": <path relative to the folder>}}`. Every passage of a file uses the same `source_id`.
6. **Report.** Summarize the files indexed, `passages_added`, passages skipped as duplicates, and the final `total_passages`. List any files you skipped and why.
