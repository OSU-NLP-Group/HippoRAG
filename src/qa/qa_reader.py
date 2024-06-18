import sys

sys.path.append('..')

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_util import init_langchain_model

from src.baselines.ircot import parse_prompt
from src.qa.hotpotqa_evaluation import update_answer
from src.qa.musique_evaluation import evaluate
from src.qa.twowikimultihopqa_evaluation import exact_match_score, f1_score

import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from tqdm import tqdm


def remove_newlines_after_first(s):
    first_newline_pos = s.find('\n')
    if first_newline_pos == -1:
        return s
    part_before_first_newline = s[:first_newline_pos + 1]
    part_after_first_newline = s[first_newline_pos + 1:].replace('\n', '')
    return part_before_first_newline + part_after_first_newline


cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
                          'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                          'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.')
cot_system_instruction_no_doc = ('As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.')


def qa_read(query: str, passages: list, few_shot: list, client):
    """

    @param query: query str
    @param passages: list of passages
    @param few_shot: few-shot in-context examples
    @param client: Langchain client
    @return: answer from passages
    """

    instruction = cot_system_instruction if len(passages) else cot_system_instruction_no_doc
    messages = [SystemMessage(instruction)]
    if few_shot:
        for sample in few_shot:
            if 'document' in sample:  # document and question from user
                cur_sample = f'{sample["document"]}\n\nQuestion: {sample["question"]}'
            else:  # no document, only question from user
                cur_sample = f'Question: {sample["question"]}'
            if 'thought' in sample:  # Chain-of-Thought
                messages.append(HumanMessage(cur_sample + '\nThought: '))
                messages.append(AIMessage(f'{sample["thought"]}\nAnswer: {sample["answer"]}'))
            else:  # No Chain-of-Thought, directly answer the question
                messages.append(HumanMessage(cur_sample + '\nAnswer: '))
                messages.append(AIMessage(f'Answer: {sample["answer"]}'))

    user_prompt = ''
    for passage in passages:
        user_prompt += f'Wikipedia Title: {passage}\n\n'
    user_prompt += 'Question: ' + query + '\nThought: '
    messages.append(HumanMessage(user_prompt))

    if few_shot:
        assert len(messages) == len(few_shot) * 2 + 2
    else:
        assert len(messages) == 2
    messages = ChatPromptTemplate.from_messages(messages).format_prompt()
    try:
        chat_completion = client.invoke(messages.to_messages())
        response_content = chat_completion.content
    except Exception as e:
        print('QA read exception', e)
        return ''
    return response_content


def parallel_qa_read(data: list, demos: list, args, client, output_path: str, total_metrics: dict, sample_id_set: set):
    def process_sample(sample):
        sample_idx, sample = sample
        sample_id = sample['_id'] if '_id' in sample else sample['id']
        if sample_id in sample_id_set:
            return None  # Skip processing if sample already processed
        query = sample['question']
        if 'retrieved' in sample:
            retrieved = sample['retrieved'][:args.num_doc]
        elif 'retrieved_id' in sample:
            retrieved = [corpus[doc_id] for doc_id in sample['retrieved_id']][:args.num_doc]
        else:
            retrieved = []
        assert len(retrieved) == args.num_doc, f'sample {sample_id}: #retrieved {len(retrieved)} != args.num_doc {args.num_doc}'
        if len(retrieved):
            if isinstance(retrieved[0], dict):
                retrieved = [item['title'] + '\n' + item['text'] for item in retrieved]
            elif isinstance(retrieved[0], list):
                retrieved = ['\n'.join(item) for item in retrieved]

        if args.dataset == 'hotpotqa':
            retrieved = [remove_newlines_after_first(item) for item in retrieved]

        response = qa_read(query, retrieved, demos, client)
        try:
            pred_ans = response.split('Answer:')[1].strip()
        except Exception as e:
            print('Parsing prediction:', e, response)
            pred_ans = response

        gold_ans = sample['answer']
        if args.dataset == 'hotpotqa':
            em, f1, precision, recall = update_answer({'em': 0, 'f1': 0, 'precision': 0, 'recall': 0}, pred_ans, gold_ans)
            return sample_idx, sample_id, retrieved, pred_ans, {'em': em, 'f1': f1, 'precision': precision, 'recall': recall}
        elif args.dataset == 'musique':
            em, f1 = evaluate({'predicted_answer': pred_ans}, sample)
            return sample_idx, sample_id, retrieved, pred_ans, {'em': em, 'f1': f1}
        elif args.dataset == '2wikimultihopqa':
            em = 1 if exact_match_score(pred_ans, gold_ans) else 0
            f1, precision, recall = f1_score(pred_ans, gold_ans)
            return sample_idx, sample_id, retrieved, pred_ans, {'em': em, 'f1': f1, 'precision': precision, 'recall': recall}

    with ThreadPoolExecutor(max_workers=args.thread) as executor:
        futures = [executor.submit(process_sample, (sample_idx, sample)) for sample_idx, sample in enumerate(data)]
        for future in tqdm(as_completed(futures), total=len(futures), desc='QA read'):
            result = future.result()
            if result is not None:
                sample_idx, sample_id, retrieved, pred_ans, metrics = result
                sample_id_set.add(sample_id)
                sample = data[sample_idx]
                sample['retrieved'] = retrieved
                sample['prediction'] = pred_ans
                for key in metrics:
                    sample['qa_' + key] = metrics[key]
                    total_metrics['qa_' + key] += metrics[key]

                if sample_idx % 50 == 0:
                    with open(output_path, 'w') as f:
                        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='retrieval results or QA reading results', choices=['hotpotqa', 'musique', '2wikimultihopqa'], required=True)
    parser.add_argument('--data', type=str, help='retrieval results or QA reading results')
    parser.add_argument('--retriever', type=str, help='retriever name to distinguish different experiments')
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    parser.add_argument('--num_demo', type=int, default=1, help='the number of few-shot examples')
    parser.add_argument('--num_doc', type=int, default=5, help='the number of in-context documents')
    parser.add_argument('--thread', type=int, default=8, help='the number of workers for parallel processing')
    args = parser.parse_args()

    output_path = f'exp/qa_{args.dataset}_{args.retriever}_{args.llm_model}_demo_{args.num_demo}_doc_{args.num_doc}.json'
    processed_id_set = set()
    total_metrics = {'qa_em': 0, 'qa_f1': 0, 'qa_precision': 0, 'qa_recall': 0}
    if args.data:
        data = json.load(open(args.data, 'r'))
    else:
        print('Please provide the retrieval results')
        exit(1)

    if args.retriever == 'none':
        args.num_doc = 0

    if args.num_doc == 0:
        if args.dataset == 'hotpotqa':
            prompt_path = 'data/ircot_prompts/hotpotqa/no_context_cot_qa_codex.txt'
            data = json.load(open('data/hotpotqa.json', 'r'))
        elif args.dataset == 'musique':
            prompt_path = 'data/ircot_prompts/musique/no_context_cot_qa_codex.txt'
            data = json.load(open('data/musique.json', 'r'))
        elif args.dataset == '2wikimultihopqa':
            prompt_path = 'data/ircot_prompts/2wikimultihopqa/no_context_cot_qa_codex.txt'
            data = json.load(open('data/2wikimultihopqa.json', 'r'))
        demos = parse_prompt(prompt_path, False)
    else:
        if os.path.isfile(output_path):  # resume from previous results
            data = json.load(open(output_path, 'r'))
            for key in total_metrics.keys():
                total_metrics[key] = sum([sample[key] for sample in data if key in sample])
        if args.dataset == 'hotpotqa':
            prompt_path = 'data/ircot_prompts/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt'
            corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
        elif args.dataset == 'musique':
            prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
            corpus = json.load(open('data/musique_corpus.json', 'r'))
        elif args.dataset == '2wikimultihopqa':
            prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
            corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        demos = parse_prompt(prompt_path)

    # processed id set
    if args.dataset in ['hotpotqa', '2wikimultihopqa']:
        processed_id_set = {sample['_id'] for sample in data if 'prediction' in sample}
    elif args.dataset in ['musique']:
        processed_id_set = {sample['id'] for sample in data if 'prediction' in sample}

    assert data and len(data)
    demos = demos[:args.num_demo]
    client = init_langchain_model(args.llm, args.llm_model)
    parallel_qa_read(data, demos, args, client, output_path, total_metrics, processed_id_set)
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print('QA results saved to', output_path)

    metric_str = ' '.join([f'{key}: {total_metrics[key] / len(data):.4f}' for key in total_metrics.keys()])
    print(metric_str)
