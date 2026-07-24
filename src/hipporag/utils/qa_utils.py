from ..prompts.prompt_template_manager import PromptTemplateManager
from .logging_utils import get_logger
from ..llm import BaseLLM

logger = get_logger(__name__)



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


def reason_step(dataset: str, prompt_template_manager: PromptTemplateManager, query: str, passages: list, thoughts: list, llm_client: BaseLLM) -> str:
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
    :return: next thought
    """

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    
    messages = prompt_template_manager.render(name=f'ircot_{dataset}', prompt_user=prompt_user)

    response_content = llm_client.infer(messages)[0]
    if not isinstance(response_content, str):
        raise TypeError(f"IRCoT reasoning expected a string response, got {type(response_content).__name__}.")
    return response_content
