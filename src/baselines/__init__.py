import torch


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_file_name(path):
    return path.split('/')[-1].replace('.jsonl', '').replace('.json', '')


def mean_pooling_embedding(input_str: str, tokenizer, model, device='cuda'):
    inputs = tokenizer(input_str, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**inputs)

    embedding = mean_pooling(outputs[0], inputs['attention_mask']).to('cpu').detach().numpy()
    return embedding


def mean_pooling_embedding_with_normalization(input_str, tokenizer, model, device='cuda'):
    encoding = tokenizer(input_str, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = mean_pooling(outputs[0], attention_mask)
    embeddings = embeddings.T.divide(torch.linalg.norm(embeddings, dim=1)).T

    return embeddings
