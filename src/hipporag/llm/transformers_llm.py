import os
from typing import List, Tuple
from copy import deepcopy
import sqlite3
import json
import time
import hashlib
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from filelock import FileLock

from .base import BaseLLM, LLMConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger

def convert_text_chat_messages_to_input_ids(messages: List[TextChatMessage], tokenizer, add_assistant_header=True) -> str:
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        chat_template=None,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        documents=None,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    return input_ids


logger = get_logger(__name__)


class LLM_Cache:
    def __init__(self, cache_dir: str, cache_filename):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_filepath =  os.path.join(cache_dir, f"{cache_filename}.sqlite")
        self.lock_file = self.cache_filepath + ".lock"

        self.__db_operation("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                message TEXT,
                metadata TEXT
            )
        """, commit=True)
    
    def __db_operation(self, sql, parameters=(), commit=False, fetchone=False):
        with FileLock(self.lock_file):
            conn = sqlite3.connect(self.cache_filepath)
            c = conn.cursor()
            c.execute(sql, parameters)
            if commit:
                conn.commit()
            if fetchone:
                row = c.fetchone()
            conn.close()
            if fetchone:
                return row

    def __params_to_key(self, params):
        key_str = f"Model: {params['model']}, Temperature: {params['temperature']}, Messages: {params['messages']}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def read(self, params):
        key = self.__params_to_key(params)
        row = self.__db_operation("SELECT message, metadata FROM cache WHERE key = ?", (key,), fetchone=True)
        if row is None:
            return None
        message, metadata_str = row
        metadata = json.loads(metadata_str)
        return message, metadata

    def write(self, params, message, metadata):
        key = self.__params_to_key(params)
        metadata_str = json.dumps(metadata)
        self.__db_operation("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)", (key, message, metadata_str), commit=True)


class TransformersLLM(BaseLLM):
    """
    To select this implementation you can initialise HippoRAG with:
        llm_model_name="meta-llama/Llama-3.1-8B-Instruct" or any other Transformer Model-ID
    """
    def __init__(self, global_config = None):
        self.global_config = global_config
        super().__init__(global_config)
        self._init_llm_config()

        self.cache = LLM_Cache(
            os.path.join(global_config.save_dir, "llm_cache"),
            self.llm_name.replace('/', '_'))
        self.model = AutoModelForCausalLM.from_pretrained(self.global_config.llm_name, device_map='auto', torch_dtype = torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.global_config.llm_name)

        self.retry = 5
        
        logger.info(f"[TransformersLLM] Model-ID: {self.global_config.llm_name}, Cache: {self.cache.cache_filepath}")

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__
        config_dict['llm_name'] = self.global_config.llm_name[len("Transformers/"):]
        config_dict['generate_params'] = {
                "n": 1,
                "temperature": config_dict.get("temperature", 0.0),
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.info(f"[TransformersLLM] Config: {self.llm_config}")

    def __llm_call(self, params):
        inputs = params["prompt_text"].to(self.model.device)
        response = self.model.generate(inputs, max_new_tokens=params.get("max_tokens", 200))
        return response
    
    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["model"] = self.global_config.llm_name
        params["messages"] = messages
        params["prompt_text"] = convert_text_chat_messages_to_input_ids(messages, self.tokenizer)
        
        cache_lookup = self.cache.read(params)
        if cache_lookup is not None:
            cached = True
            message, metadata = cache_lookup
        else:
            cached = False
            response = self.__llm_call(params)
            message = self.tokenizer.decode(response[0], skip_special_tokens=True)
            metadata = {
                "prompt_tokens": params["prompt_text"].shape[1], 
                "completion_tokens": response.shape[1],
            }
            self.cache.write(params, message, metadata)

        return message, metadata, cached
