def chunk_corpus(corpus: list, chunk_size: int = 64) -> list:
    """
    Chunk the corpus into smaller parts. Run the following command to download the required nltk data:
    python -c "import nltk; nltk.download('punkt')"

    @param corpus: the formatted corpus, see README.md
    @param chunk_size: the size of each chunk, i.e., the number of words in each chunk
    @return: chunked corpus, a list
    """
    from nltk.tokenize import sent_tokenize, word_tokenize

    new_corpus = []
    for p in corpus:
        text = p['text']
        idx = p['idx'] if 'idx' in p else p['_id']
        title = p['title']

        sentences = sent_tokenize(text)
        current_chunk = []
        current_chunk_size = 0

        chunk_idx = 0
        for sentence in sentences:
            words = word_tokenize(sentence)
            if current_chunk_size + len(words) > chunk_size:
                new_corpus.append({
                    **p,
                    'title': title,
                    'text': " ".join(current_chunk),
                    'idx': idx + f"_{chunk_idx}",
                })
                current_chunk = words
                current_chunk_size = len(words)
                chunk_idx += 1
            else:
                current_chunk.extend(words)
                current_chunk_size += len(words)

        if current_chunk:  # there are still some words left
            new_corpus.append({
                **p,
                'title': title,
                'text': " ".join(current_chunk),
                'idx': f"{idx}_{chunk_idx}",
            })

    return new_corpus


def merge_chunk_scores(id_score: dict):
    """
    Merge the scores of chunks into the original passage
    @param id_score: a dictionary of passage_id (the chunk id, str) -> score (float)
    @return: a merged dictionary of passage_id (the original passage id, str) -> score (float)
    """
    merged_scores = {}
    for passage_id, score in id_score.items():
        passage_id = passage_id.split('_')[0]
        if passage_id not in merged_scores:
            merged_scores[passage_id] = 0
        merged_scores[passage_id] = max(merged_scores[passage_id], score)
    return merged_scores
