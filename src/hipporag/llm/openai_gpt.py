import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from packaging import version

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, **config_dict)

    def __init__(self, cache_dir, cache_filename: str = None,
                 llm_name: str = "gpt-4o-mini", api_key: str = None, llm_base_url: str = None, 
                 high_throughput: bool = True,
                 **kwargs) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        self.llm_name = llm_name
        self.llm_base_url = llm_base_url
        self._init_llm_config(**kwargs)
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
        else:
            client = None
        self.openai_client = OpenAI(base_url=self.llm_base_url, api_key=api_key, http_client=client)

    def _init_llm_config(self, **kwargs) -> None:
        config_dict = {
            "llm_name": self.llm_name,
            "llm_base_url": self.llm_base_url,
            "generate_params": {
                "model": self.llm_name,
                "max_completion_tokens": kwargs.get("max_new_tokens", 400),
                "n": kwargs.get("num_gen_choices", 1),
                "seed": kwargs.get("seed", 0),
                "temperature": kwargs.get("temperature", 0.0),
            }
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')

        response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata


