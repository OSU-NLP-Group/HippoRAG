from typing import Tuple, List
import torch
import torch.cuda
import os
import json

from .base import BaseLLM, LLMConfig
from ..utils.llm_utils import TextChatMessage, PROMPT_JSON_TEMPLATE
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers import BitsAndBytesConfig
import gc

def convert_text_chat_messages_to_strings(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer, add_assistant_header=True) -> List[str]:
    return tokenizer.apply_chat_template(conversation=messages, tokenize=False)

def convert_text_chat_messages_to_input_ids(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer, add_assistant_header=True) -> List[int]:
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        chat_template=None,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        documents=None,
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    return encoded['input_ids'][0].tolist()

class TransformersOffline:

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig()

    def __init__(self, global_config, cache_dir=None, cache_filename=None, max_model_len=4096, **kwargs):
        model_name = kwargs.get('model_name', global_config.llm_name)
        if model_name is None:
            model_name = 'meta-llama/Llama-3.3-70B-Instruct'
        
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup quantization if needed
        quantization_config = None
        if 'bnb' in model_name or kwargs.get('quantization') == 'bitsandbytes':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            'torch_dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            'device_map': 'auto',
            'trust_remote_code': True,
        }
        
        if quantization_config is not None:
            model_kwargs['quantization_config'] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        
        # Setup cache
        if cache_filename is None:
            cache_filename = f'{model_name.replace("/", "_")}_cache.sqlite'
        if cache_dir is None:
            cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file_name = os.path.join(cache_dir, cache_filename)
    
    def _generate_response(self, input_ids: torch.Tensor, max_tokens: int = 2048) -> Tuple[str, dict]:
        """Generate response for a single input"""
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            # Generate
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,  # temperature=0 equivalent
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            
            # Extract only the new tokens (response)
            response_ids = outputs[0][input_ids.shape[1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Calculate tokens
            prompt_tokens = input_ids.shape[1]
            completion_tokens = len(response_ids)
            
            metadata = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
            
            return response, metadata

    def _generate_batch_response(self, batch_input_ids: List[torch.Tensor], max_tokens: int = 2048) -> Tuple[List[str], dict]:
        """Generate response for a batch of inputs"""
        all_responses = []
        all_prompt_tokens = []
        all_completion_tokens = []
        
        # Process each input separately for now (can be optimized for true batching later)
        for input_ids in batch_input_ids:
            response, metadata = self._generate_response(input_ids.unsqueeze(0), max_tokens)
            all_responses.append(response)
            all_prompt_tokens.append(metadata["prompt_tokens"])
            all_completion_tokens.append(metadata["completion_tokens"])
            
            # Clear GPU cache to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        metadata = {
            "prompt_tokens": sum(all_prompt_tokens),
            "completion_tokens": sum(all_completion_tokens),
            "num_request": len(batch_input_ids)
        }
        
        return all_responses, metadata

    def infer(self, messages: List[TextChatMessage], max_tokens=2048):
        logger.info(f"Calling Transformers offline, # of messages {len(messages)}")
        
        # Convert messages to input_ids
        input_ids_list = convert_text_chat_messages_to_input_ids(messages, self.tokenizer)
        input_ids = torch.tensor([input_ids_list])
        
        response, metadata = self._generate_response(input_ids, max_tokens)
        
        return response, metadata

    def batch_infer(self, messages_list: List[List[TextChatMessage]], max_tokens=2048, json_template=None):
        if len(messages_list) > 1:
            logger.info(f"Calling Transformers offline, # of messages {len(messages_list)}")
        
        # Note: json_template guided generation is not easily implemented in transformers
        # This is a limitation compared to VLLM. For now, we'll log a warning if it's used.
        if json_template is not None:
            logger.warning("JSON template guided generation is not implemented in TransformersOffline. Ignoring json_template parameter.")
        
        # Convert all messages to input_ids
        all_input_ids = []
        for messages in messages_list:
            input_ids_list = convert_text_chat_messages_to_input_ids(messages, self.tokenizer)
            all_input_ids.append(torch.tensor(input_ids_list))
        
        all_responses, metadata = self._generate_batch_response(all_input_ids, max_tokens)
        
        return all_responses, metadata

    def get_tokenizer(self):
        """Get the tokenizer (for compatibility with VLLM interface)"""
        return self.tokenizer
