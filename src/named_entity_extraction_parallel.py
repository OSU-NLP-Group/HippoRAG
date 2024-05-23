import argparse
import os.path
from together import Together
from processing import *
from openai import OpenAI
from tqdm import tqdm

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

query_prompt_template = """
Question: {}

"""

def named_entity_recognition(text: str, model_name='gpt-3.5-turbo-1106'):

    messages = [{'role': 'system', 'content': "You're a very effective entity extraction system."}]
    messages.append({'role': 'user', 'content': query_prompt_one_shot_input})
    messages.append({'role': 'assistant', 'content': query_prompt_one_shot_output})
    messages.append({'role': 'user', 'content': query_prompt_template.format(text)})
    # try:

    if 'gpt' in model_name:
        chat_completion = client.chat.completions.create(messages=messages, model=model_name, temperature=0, max_tokens=300, stop=['\n\n'], response_format={"type": "json_object"})
        response_content = chat_completion.choices[0].message.content
    else:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=300,
            stop=['\n\n']
        )
        res = chat_completion.choices[0].message.content
        response_content = extract_json_dict(res)
        try:
            assert 'named_entities' in response_content
            response_content = str(response_content)
        except:
            print('ERROR')
            response_content = {'named_entities':[]}

    total_tokens = chat_completion.usage.total_tokens
    # except:
    #     print(text)
    #     return '',0

    return response_content, total_tokens

def run_ner_on_texts(texts):
    ner_output = []
    total_cost = 0

    for text in tqdm(texts):
        ner, cost = named_entity_recognition(text, model_name)
        ner_output.append(ner)
        total_cost += cost

    return ner_output, total_cost

import sys

if __name__ == '__main__':
    # Get the first argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name',type=str)

    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model_name

    output_file = 'output/{}_queries.named_entity_output.tsv'.format(dataset)

    if 'gpt' in model_name:
        client = OpenAI()
    else:
        # set TOGETHER_API_KEY environment variable before running this function
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    ## Extract Entities from Queries

    try:
        queries_df = pd.read_json(f'data/{dataset}.json')

        if 'hotpotqa' in dataset:
            queries_df = queries_df[['question']]
            queries_df['0'] = queries_df['question']
            queries_df['query'] = queries_df['question']
            query_name = 'query'
        else:
            query_name = 'question'

        try:
            output_df = pd.read_csv(output_file, sep='\t')
        except:
            output_df = []

        if len(queries_df) != len(output_df):
            queries = queries_df[query_name].values

            num_processes = 10

            splits = np.array_split(range(len(queries)), num_processes)

            args = []

            for split in splits:
                args.append([queries[i] for i in split])

            with Pool(processes=num_processes) as pool:
                outputs = pool.map(run_ner_on_texts, args)

            chatgpt_total_tokens = 0

            query_triples = []

            for output in outputs:
                query_triples.extend(output[0])
                chatgpt_total_tokens += output[1]

            current_cost = 0.002 * chatgpt_total_tokens / 1000

            queries_df['triples'] = query_triples
            queries_df.to_csv(output_file,sep='\t')
        else:
            pass
    except:
        print('No queries will be processed for later retrieval.')