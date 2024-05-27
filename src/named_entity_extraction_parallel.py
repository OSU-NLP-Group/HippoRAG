import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from processing import *
from tqdm import tqdm

from src.langchain_util import init_langchain_model

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


def named_entity_recognition(text: str):
    query_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage("You're a very effective entity extraction system."),
                                                          HumanMessage(query_prompt_one_shot_input),
                                                          AIMessage(query_prompt_one_shot_output),
                                                          HumanMessage(query_prompt_template.format(text))])
    query_ner_messages = query_ner_prompts.format_prompt()

    if isinstance(client, ChatOpenAI):  # JSON mode
        chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=300, stop=['\n\n'], response_format={"type": "json_object"})
        response_content = chat_completion.content
    else:  # no JSON mode
        chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=300, stop=['\n\n'])
        response_content = chat_completion.content
        response_content = extract_json_dict(response_content)

        try:
            assert 'named_entities' in response_content
            response_content = str(response_content)
        except Exception as e:
            print('Query NER exception', e)
            response_content = {'named_entities': []}

    total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
    return response_content, total_tokens


def run_ner_on_texts(texts):
    ner_output = []
    total_cost = 0

    for text in tqdm(texts):
        ner, cost = named_entity_recognition(text)
        ner_output.append(ner)
        total_cost += cost

    return ner_output, total_cost


if __name__ == '__main__':
    # Get the first argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    parser.add_argument('--num_processes', type=int, default=10, help='Number of processes')

    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model_name

    output_file = 'output/{}_queries.named_entity_output.tsv'.format(dataset)

    client = init_langchain_model(args.llm, model_name)  # LangChain model

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

            num_processes = args.num_processes

            splits = np.array_split(range(len(queries)), num_processes)

            args = []

            for split in splits:
                args.append([queries[i] for i in split])

            if num_processes == 1:
                outputs = [run_ner_on_texts(args[0])]
            else:
                with Pool(processes=num_processes) as pool:
                    outputs = pool.map(run_ner_on_texts, args)

            chatgpt_total_tokens = 0

            query_triples = []

            for output in outputs:
                query_triples.extend(output[0])
                chatgpt_total_tokens += output[1]

            current_cost = 0.002 * chatgpt_total_tokens / 1000

            queries_df['triples'] = query_triples
            queries_df.to_csv(output_file, sep='\t')
        else:
            pass
    except Exception as e:
        print('No queries will be processed for later retrieval.', e)
