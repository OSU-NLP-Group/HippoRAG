from typing import Tuple, List
import torch.cuda
import outlines.generate as generate
import outlines.models as models
import json

from .base import BaseLLM, LLMConfig
from ..utils.llm_utils import TextChatMessage, get_pydantic_model
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

from transformers import PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer

def convert_text_chat_messages_to_strings(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer, add_assistant_header=True) -> List[str]:
    return tokenizer.apply_chat_template(conversation=messages, tokenize=False)

def convert_text_chat_messages_to_input_string(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer, add_assistant_header=True) -> str:
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        chat_template=None,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        documents=None,
    )
    return prompt

from vllm import SamplingParams
class TransformersOffline:

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig()

    def __init__(self, global_config, cache_dir=None, cache_filename=None, max_model_len=4096, **kwargs):
        model_name = kwargs.get('model_name', global_config.llm_name)
        if model_name is None:
            model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        import os
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype = torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if cache_filename is None:
            cache_filename = f'{model_name.replace("/", "_")}_cache.sqlite'
        if cache_dir is None:
            cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        self.cache_file_name = os.path.join(cache_dir, cache_filename)
    
    def infer(self, messages: List[TextChatMessage], max_tokens=2048):
        logger.info(f"Calling Transformers offline, # of messages {len(messages)}")
        messages_list = [messages]
        prompt_text = convert_text_chat_messages_to_input_string(messages_list, self.tokenizer)
        input_ids = self.tokenizer.encode(prompt_text)
        transformers_output = self.model.generate(input_ids, max_new_tokens=max_tokens)
        response = transformers_output[0]['generated_text']
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        completion_tokens = len(self.tokenizer.encode(response))
        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        return response, metadata

    def batch_infer(self, messages_list: List[List[TextChatMessage]], max_tokens=2048, json_template=None):
        if len(messages_list) > 1:
            logger.info(f"Calling Transformers offline, # of messages {len(messages_list)}, using batchsize = 4")

        all_prompt_texts = [convert_text_chat_messages_to_input_string(messages, self.tokenizer) for messages in messages_list]

        guided = None
        if json_template is not None:
            guided_json=get_pydantic_model(json_template)
            outlines_model = models.Transformers(self.model, self.tokenizer)
            generator = generate.json(outlines_model, guided_json)
            transformers_outputs = []
            for i in range(0, len(all_prompt_texts), 4):
                transformers_output = generator(all_prompt_texts[i:i+4], max_tokens=max_tokens)
                transformers_outputs.extend(transformers_output)
        else:
            transformers_outputs = []
            for i in range(0, len(all_prompt_texts), 4):
                transformers_output = self.model.generate(all_prompt_texts[i:i+4], max_tokens=max_tokens)
                transformers_outputs.extend(transformers_output)
        all_responses = [completion.model_dump_json() for completion in transformers_outputs]
        all_prompt_tokens = [len(self.tokenizer.encode(prompt)) for prompt in all_prompt_texts]
        all_completion_tokens = [len(self.tokenizer.encode(response)) for response in all_responses]

        metadata = {
            "prompt_tokens": sum(all_prompt_tokens),
            "completion_tokens": sum(all_completion_tokens),
            "num_request": len(messages_list)
        }
        return all_responses, metadata
