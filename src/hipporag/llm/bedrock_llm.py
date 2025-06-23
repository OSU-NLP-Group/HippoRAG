import os
from typing import List, Tuple
from copy import deepcopy
import sqlite3
import json
import time
import hashlib

import litellm
from filelock import FileLock

from .base import BaseLLM, LLMConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger


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


class BedrockLLM(BaseLLM):
    """
    To select this implementation you can initialise HippoRAG with:
        llm_model_name="anthropic.claude-3-5-haiku-20241022-v1:0" or any other Bedrock Model-ID
    """
    def __init__(self, global_config = None):
        self.global_config = global_config
        super().__init__(global_config)
        self._init_llm_config()

        self.cache = LLM_Cache(
            os.path.join(global_config.save_dir, "llm_cache"),
            self.llm_name.replace('/', '_'))        
        
        self.retry = 5
        
        logger.info(f"[BedrockLLM] Model-ID: {self.global_config.llm_name}, Cache: {self.cache.cache_filepath}")

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__
        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "n": 1,
                "temperature": config_dict.get("temperature", 0.0),
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.info(f"[BedrockLLM] Config: {self.llm_config}")

    def __llm_call(self, params):
        num, wait_s = 0, 0.5
        while True:
            try:
                return litellm.completion(**params)
            except Exception as e:
                num += 1
                if num > self.retry:
                    raise e
                
                logger.warning(f"Bedrock LLM Exception: {e}\nRetry #{num} after {wait_s} seconds")
                time.sleep(wait_s)
                wait_s *= 2
    
    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        
        cache_lookup = self.cache.read(params)
        if cache_lookup is not None:
            cached = True
            message, metadata = cache_lookup
        else:
            cached = False
            response = self.__llm_call(params)
            message = response.choices[0].message.content
            metadata = {
                "prompt_tokens": response.usage.prompt_tokens, 
                "completion_tokens": response.usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
            }
            self.cache.write(params, message, metadata)

        return message, metadata, cached
