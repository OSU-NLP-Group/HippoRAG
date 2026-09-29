import functools
import hashlib
import json
import os
import sqlite3
from typing import List, Tuple

from filelock import FileLock
from openai import AzureOpenAI, DefaultHttpxClient, OpenAI

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from ..utils.openai_utils import resolve_azure_openai_settings, resolve_openai_api_key, validate_openai_base_url
from .base import BaseLLM, LLMConfig, normalize_generation_token_params
from .gateways import chat_max_tokens_key

logger = get_logger(__name__)


def _validate_azure_request_model(global_config, request_model_name, params) -> None:
    if getattr(global_config, "azure_endpoint", None) is None:
        return
    if params.get("model") != request_model_name:
        raise ValueError(f"Azure request model must match the configured deployment {request_model_name!r}.")
    extra_body = params.get("extra_body")
    if isinstance(extra_body, dict) and "model" in extra_body:
        raise ValueError("Azure extra_body must not override the configured deployment model.")

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        global_config = getattr(self, "global_config", None)
        request_model_name = getattr(self, "request_model_name", None)
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # Include every generation parameter because any of them can change the response.
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        key_data = normalize_generation_token_params(gen_params, kwargs, "_max_tokens")
        _validate_azure_request_model(global_config, request_model_name, key_data)
        key_data["messages"] = messages
        key_data["_cache_schema"] = 3
        key_data["_provider_class"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        key_data["_endpoint"] = getattr(global_config, "azure_endpoint", None) or getattr(global_config, "llm_base_url", None)
        key_data["_azure_api_version"] = getattr(global_config, "azure_api_version", None)
        key_data["_azure_chat_deployment"] = getattr(global_config, "azure_chat_deployment", None)
        if key_data.get("store") is True:
            message, metadata = func(self, *args, **kwargs)
            return message, metadata, False
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        database_lock_file = self.cache_file_name + ".lock"
        key_lock_file = f"{self.cache_file_name}.{key_hash[:2]}.lock"
        with FileLock(key_lock_file):
            with FileLock(database_lock_file):
                with sqlite3.connect(self.cache_file_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS cache (
                            key TEXT PRIMARY KEY,
                            message TEXT,
                            metadata TEXT
                        )
                    """)
                    cursor.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                    row = cursor.fetchone()
            if row is not None:
                message, metadata_str = row
                return message, json.loads(metadata_str), True

            message, metadata = func(self, *args, **kwargs)

            with FileLock(database_lock_file):
                with sqlite3.connect(self.cache_file_name) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                        (key_hash, message, json.dumps(metadata)),
                    )

        return message, metadata, False

    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config, max_retries=global_config.max_retry_attempts)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__(global_config)
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self.max_retries = kwargs.get("max_retries", 2)
        azure_settings = None
        self.request_model_name = self.llm_name
        if self.global_config.azure_endpoint is not None:
            azure_settings = resolve_azure_openai_settings(
                self.global_config.azure_endpoint,
                api_version=self.global_config.azure_api_version,
                deployment=self.global_config.azure_chat_deployment,
                operation="chat.completions",
            )
            self.request_model_name = azure_settings.deployment or self.llm_name
        else:
            self.llm_base_url = validate_openai_base_url(self.llm_base_url, "chat/completions", "llm_base_url")
        self._init_llm_config()

        client = DefaultHttpxClient(timeout=5 * 60) if high_throughput else None
        try:
            if azure_settings is None:
                self.openai_client = OpenAI(
                    api_key=resolve_openai_api_key(self.llm_base_url, self.global_config.llm_api_key_env, "llm_api_key_env"),
                    base_url=self.llm_base_url,
                    http_client=client,
                    max_retries=self.max_retries,
                )
            else:
                self.openai_client = AzureOpenAI(
                    api_key=resolve_openai_api_key(None, self.global_config.llm_api_key_env, "llm_api_key_env"),
                    api_version=azure_settings.api_version,
                    azure_endpoint=azure_settings.endpoint,
                    azure_deployment=azure_settings.deployment,
                    http_client=client,
                    max_retries=self.max_retries,
                )
        except Exception:
            if client is not None:
                client.close()
            raise

    def _init_llm_config(self) -> None:
        generate_params = {
            "model": self.request_model_name,
        }
        if self.global_config.max_new_tokens is not None:
            generate_params["max_completion_tokens"] = self.global_config.max_new_tokens
        if self.global_config.seed is not None:
            generate_params["seed"] = self.global_config.seed
        if self.global_config.temperature is not None:
            generate_params["temperature"] = self.global_config.temperature
        config_dict = {
            'llm_name': self.global_config.llm_name,
            'llm_base_url': self.global_config.llm_base_url,
            'generate_params': generate_params,
        }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[str, dict, bool]:
        target_token_key = chat_max_tokens_key(self.global_config, self.global_config.llm_base_url)
        params = normalize_generation_token_params(self.llm_config.generate_params, kwargs, target_token_key)
        _validate_azure_request_model(self.global_config, getattr(self, "request_model_name", getattr(self, "llm_name", None)), params)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        response = self.openai_client.chat.completions.create(**params)
        if len(response.choices) != 1:
            raise ValueError(f"HippoRAG expected exactly one OpenAI choice, received {len(response.choices)}.")
        choice = response.choices[0]
        response_message = choice.message.content
        if not isinstance(response_message, str) or not response_message:
            refusal = getattr(choice.message, "refusal", None)
            detail = f" Refusal: {refusal}" if refusal else ""
            raise ValueError(f"OpenAI response did not contain non-empty text.{detail}")
        usage = response.usage
        if usage is None:
            raise ValueError("OpenAI response omitted usage; HippoRAG cannot account for this request safely.")
        total_tokens = getattr(usage, "total_tokens", None)
        metadata = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": total_tokens if total_tokens is not None else usage.prompt_tokens + usage.completion_tokens,
            "finish_reason": choice.finish_reason,
        }
        for key, value in (("response_id", getattr(response, "id", None)), ("model", getattr(response, "model", None)), ("request_id", getattr(response, "_request_id", None))):
            if value is not None:
                metadata[key] = value
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        if cached_tokens is not None:
            metadata["cached_tokens"] = cached_tokens
        if reasoning_tokens is not None:
            metadata["reasoning_tokens"] = reasoning_tokens

        return response_message, metadata

    def close(self) -> None:
        self.openai_client.close()
