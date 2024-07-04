import sys

sys.path.append('.')

import ipdb
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_util import init_langchain_model
from transformers.hf_argparser import string_to_bool
import argparse
import json

from tqdm import tqdm

from hipporag import HippoRAG

ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'


def parse_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content by the metadata pattern
    parts = content.split('# METADATA: ')
    parsed_data = []

    for part in parts[1:]:  # Skip the first split as it will be empty
        metadata_section, rest_of_data = part.split('\n', 1)
        metadata = json.loads(metadata_section)
        document_sections = rest_of_data.strip().split('\n\nQ: ')
        document_text = document_sections[0].strip()
        qa_pair = document_sections[1].split('\nA: ')
        question = qa_pair[0].strip()
        answer = qa_pair[1].strip()

        parsed_data.append({
            'metadata': metadata,
            'document': document_text,
            'question': question,
            'answer': answer
        })

    return parsed_data


def retrieve_step(query: str, corpus, top_k: int, rag: HippoRAG, dataset: str):
    ranks, scores, logs = rag.rank_docs(query, top_k=top_k)
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        retrieved_passages = []
        for rank in ranks:
            key = list(corpus.keys())[rank]
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    else:
        retrieved_passages = [corpus[rank]['title'] + '\n' + corpus[rank]['text'] for rank in ranks]
    return retrieved_passages, scores, logs


def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
    merged_dict = {}

    # Iterate through each element in the list
    for element in elements:
        # Split the element into lines and get the first line
        lines = element.split('\n')
        first_line = lines[0]

        # Check if the first line is already a key in the dictionary
        if first_line in merged_dict:
            # Append the current element to the existing value
            merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
        else:
            # Add the current element as a new entry in the dictionary
            merged_dict[first_line] = prefix + element

    # Extract the merged elements from the dictionary
    merged_elements = list(merged_dict.values())
    return merged_elements


def reason_step(dataset, few_shot: list, query: str, passages: list, thoughts: list, client):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
    :return: next thought
    """
    prompt_demo = ''
    for sample in few_shot:
        prompt_demo += f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["answer"]}\n\n'

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    messages = ChatPromptTemplate.from_messages([SystemMessage(ircot_reason_instruction + '\n\n' + prompt_demo),
                                                 HumanMessage(prompt_user)]).format_prompt()

    try:
        chat_completion = client.invoke(messages.to_messages())
        response_content = chat_completion.content
    except Exception as e:
        print(e)
        return ''
    return response_content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--num_demo', type=int, default=1, help='the number of demo samples')
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--top_k', type=int, default=8, help='retrieving k documents at each step')
    parser.add_argument('--doc_ensemble', type=str, default='t')
    parser.add_argument('--dpr_only', type=str, default='f')
    parser.add_argument('--graph_alg', type=str, default='ppr')
    parser.add_argument('--wo_node_spec', action='store_true')
    parser.add_argument('--sim_threshold', type=float, default=0.8)
    parser.add_argument('--recognition_threshold', type=float, default=0.9)
    parser.add_argument('--damping', type=float, default=0.1)
    parser.add_argument('--force_retry', action='store_true')
    args = parser.parse_args()

    # Please set environment variable OPENAI_API_KEY
    doc_ensemble = string_to_bool(args.doc_ensemble)
    dpr_only = string_to_bool(args.dpr_only)

    client = init_langchain_model(args.llm, args.llm_model)
    llm_model_name_processed = args.llm_model.replace('/', '_').replace('.', '_')
    if args.llm_model == 'gpt-3.5-turbo-1106':  # Default OpenIE system
        colbert_configs = {'root': f'data/lm_vectors/colbert/{args.dataset}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}
    # else:
    #     colbert_configs = {'root': f'data/lm_vectors/colbert/{args.dataset}_{llm_model_name_processed}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}

    rag = HippoRAG(args.dataset, args.llm, args.llm_model, args.retriever, doc_ensemble=doc_ensemble, node_specificity=not (args.wo_node_spec), sim_threshold=args.sim_threshold,
                   colbert_config=colbert_configs, dpr_only=dpr_only, graph_alg=args.graph_alg, damping=args.damping, recognition_threshold=args.recognition_threshold)

    data = json.load(open(f'data/{args.dataset}.json', 'r'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
    max_steps = args.max_steps

    if max_steps > 1:
        if 'hotpotqa' in args.dataset:
            prompt_path = 'data/ircot_prompts/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        elif 'musique' in args.dataset:
            prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
        elif '2wikimultihopqa' in args.dataset:
            prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        else:
            prompt_path = f'data/ircot_prompts/{args.dataset}/gold_with_3_distractors_context_cot_qa_codex.txt'

        few_shot_samples = parse_prompt(prompt_path)[:args.num_demo]

    doc_ensemble_str = f'doc_ensemble_{args.recognition_threshold}' if doc_ensemble else 'no_ensemble'

    if dpr_only:
        dpr_only_str = 'dpr_only'
    else:
        dpr_only_str = 'hipporag'

    if args.graph_alg == 'ppr':
        output_path = f'output/ircot/ircot_results_{args.dataset}_{dpr_only_str}_{rag.graph_creating_retriever_name_processed}_demo_{args.num_demo}_{llm_model_name_processed}_{doc_ensemble_str}_step_{max_steps}_top_{args.top_k}_sim_thresh_{args.sim_threshold}'
        if args.damping != 0.1:
            output_path += f'_damping_{args.damping}'
    else:
        output_path = f'output/ircot/ircot_results_{args.dataset}_{dpr_only_str}_{rag.graph_creating_retriever_name_processed}_demo_{args.num_demo}_{llm_model_name_processed}_{doc_ensemble_str}_step_{max_steps}_top_{args.top_k}_{args.graph_alg}_sim_thresh_{args.sim_threshold}'

    if args.wo_node_spec:
        output_path += 'wo_node_spec'

    output_path += '.json'

    k_list = [1, 2, 5, 10, 15, 20, 30, 40, 50, 80, 100]
    total_recall = {k: 0 for k in k_list}

    force_retry = args.force_retry

    if force_retry:
        results = []
        processed_ids = set()
    else:
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
                processed_ids = {sample['_id'] for sample in results}
            else:
                processed_ids = {sample['id'] for sample in results}

            for sample in results:
                total_recall = {k: total_recall[k] + sample['recall'][str(k)] for k in k_list}
        except Exception as e:
            print(e)
            print('Results file maybe empty, cannot be loaded.')
            results = []
            processed_ids = set()

    print(f'Loaded {len(results)} results from {output_path}')
    if len(results) > 0:
        for k in k_list:
            print(f'R@{k}: {total_recall[k] / len(results):.4f} ', end='')
        print()

    for sample_idx, sample in tqdm(enumerate(data), total=len(data), desc='IRCoT retrieval'):  # for each sample
        if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
            sample_id = sample['_id']
        else:
            sample_id = sample['id']

        if sample_id in processed_ids:
            continue

        query = sample['question']
        all_logs = {}

        retrieved_passages, scores, logs = retrieve_step(query, corpus, args.top_k, rag, args.dataset)

        it = 1
        all_logs[it] = logs

        thoughts = []
        retrieved_passages_dict = {passage: score for passage, score in zip(retrieved_passages, scores)}

        while it < max_steps:  # for each iteration of IRCoT
            new_thought = reason_step(args.dataset, few_shot_samples, query, retrieved_passages[:args.top_k], thoughts, client)
            thoughts.append(new_thought)
            if 'So the answer is:' in new_thought:
                break
            it += 1

            new_retrieved_passages, new_scores, logs = retrieve_step(new_thought, corpus, args.top_k, rag, args.dataset)
            all_logs[it] = logs

            for passage, score in zip(new_retrieved_passages, new_scores):
                if passage in retrieved_passages_dict:
                    retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
                else:
                    retrieved_passages_dict[passage] = score

            retrieved_passages, scores = zip(*retrieved_passages_dict.items())

            sorted_passages_scores = sorted(zip(retrieved_passages, scores), key=lambda x: x[1], reverse=True)
            retrieved_passages, scores = zip(*sorted_passages_scores)
        # end iteration

        # calculate recall
        if args.dataset in ['hotpotqa', 'hotpotqa_train']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        elif args.dataset in ['2wikimultihopqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        else:
            gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
            gold_items = set([item['title'] + '\n' + item['text'] for item in gold_passages])
            retrieved_items = retrieved_passages

        # calculate metrics
        recall = dict()
        print(f'idx: {sample_idx + 1} ', end='')
        for k in k_list:
            recall[k] = round(sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items), 4)
            total_recall[k] += recall[k]
            print(f'R@{k}: {total_recall[k] / (sample_idx + 1):.4f} ', end='')
        print()
        print('[ITERATION]', it, '[PASSAGE]', len(retrieved_passages), '[THOUGHT]', thoughts)

        # record results
        phrases_in_gold_docs = []
        for gold_item in gold_items:
            phrases_in_gold_docs.append(rag.get_phrases_in_doc_str(gold_item))

        if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
            sample['supporting_docs'] = [item for item in sample['supporting_facts']]
        else:
            sample['supporting_docs'] = [item for item in sample['paragraphs'] if item['is_supporting']]
            del sample['paragraphs']

        sample['retrieved'] = retrieved_passages[:10]
        sample['retrieved_scores'] = scores[:10]
        sample['nodes_in_gold_doc'] = phrases_in_gold_docs
        sample['recall'] = recall
        first_log = all_logs[1]
        for key in first_log.keys():
            sample[key] = first_log[key]
        sample['thoughts'] = thoughts
        results.append(sample)

        if (sample_idx + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f)

    # save results
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f'Saved {len(results)} results to {output_path}')
