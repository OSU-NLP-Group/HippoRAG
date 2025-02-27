from typing import Tuple, List
import torch.cuda

from .base import BaseLLM, LLMConfig
from ..utils.llm_utils import TextChatMessage, PROMPT_JSON_TEMPLATE
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

from transformers import PreTrainedTokenizer

def convert_text_chat_messages_to_strings(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer, add_assistant_header=True) -> List[str]:
    return tokenizer.apply_chat_template(conversation=messages, tokenize=False)

def convert_text_chat_messages_to_input_ids(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer, add_assistant_header=True) -> List[List[int]]:
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        chat_template=None,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        documents=None,
    )
    encoded = tokenizer(prompt, add_special_tokens=False)
    return encoded['input_ids']
from vllm import SamplingParams, LLM
class VLLMOffline:

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig()

    def __init__(self, global_config, cache_dir=None, cache_filename=None, max_model_len=4096, **kwargs):
        model_name = kwargs.get('model_name', global_config.llm_name)
        if model_name is None:
            model_name = 'meta-llama/Llama-3.3-70B-Instruct'
        from vllm import LLM
        pipeline_parallel_size = 1
        tensor_parallel_size = kwargs.get('num_gpus', torch.cuda.device_count())
        if '8B' in model_name:
            tensor_parallel_size = 1
        if 'bnb' in model_name:
            kwargs['quantization'] = 'bitsandbytes'
            kwargs['load_format'] = 'bitsandbytes'
            tensor_parallel_size = 1
            pipeline_parallel_size = kwargs.get('num_gpus', torch.cuda.device_count())

        import os
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        self.model_name = model_name
        self.client = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
                          seed=kwargs.get('seed', 0), dtype='auto', max_seq_len_to_capture=max_model_len, enable_prefix_caching=True,
                          enforce_eager=False, gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.6),
                          max_model_len=max_model_len, quantization=kwargs.get('quantization', None), load_format=kwargs.get('load_format', 'auto'), trust_remote_code=True)
        
        self.tokenizer = self.client.get_tokenizer()
        if cache_filename is None:
            cache_filename = f'{model_name.replace("/", "_")}_cache.sqlite'
        if cache_dir is None:
            cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        self.cache_file_name = os.path.join(cache_dir, cache_filename)
    
    def infer(self, messages: List[TextChatMessage], max_tokens=2048):
        logger.info(f"Calling VLLM offline, # of messages {len(messages)}")
        messages_list = [messages]
        prompt_ids = convert_text_chat_messages_to_input_ids(messages_list, self.tokenizer)

        vllm_output = self.client.generate(prompt_token_ids=prompt_ids,  sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
        response = vllm_output[0].outputs[0].text
        prompt_tokens = len(vllm_output[0].prompt_token_ids)
        completion_tokens = len(vllm_output[0].outputs[0].token_ids )
        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        return response, metadata

    def batch_infer(self, messages_list: List[List[TextChatMessage]], max_tokens=2048, json_template=None):
        if len(messages_list) > 1:
            logger.info(f"Calling VLLM offline, # of messages {len(messages_list)}")

        guided = None
        if json_template is not None:
            from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest
            guided = GuidedDecodingRequest(guided_json=PROMPT_JSON_TEMPLATE[json_template])

        all_prompt_ids = [convert_text_chat_messages_to_input_ids(messages, self.tokenizer) for messages in messages_list]
        vllm_output = self.client.generate(prompt_token_ids=all_prompt_ids,
                                           sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
                                           guided_options_request=guided)

        all_responses = [completion.outputs[0].text for completion in vllm_output]
        all_prompt_tokens = [len(completion.prompt_token_ids) for completion in vllm_output]
        all_completion_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]

        metadata = {
            "prompt_tokens": sum(all_prompt_tokens),
            "completion_tokens": sum(all_completion_tokens),
            "num_request": len(messages_list)
        }
        return all_responses, metadata
