import os
from typing import List, Tuple

from openai import OpenAI

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from ..utils.openai_utils import validate_openai_base_url
from .base import BaseLLM, LLMConfig, normalize_generation_token_params
from .openai_gpt import cache_response

logger = get_logger(__name__)

DEFAULT_ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"


class AtlasCloudLLM(BaseLLM):
    """Atlas Cloud implementation using the OpenAI Chat Completions API.

    Atlas Cloud exposes an OpenAI-compatible endpoint that routes requests across
    many models. Model names use the ``vendor/model`` namespace (for example
    ``deepseek-ai/deepseek-v3.2``), so a full HippoRAG model name looks like
    ``atlascloud/deepseek-ai/deepseek-v3.2``.
    """

    prefix = "atlascloud/"

    def __init__(self, global_config: BaseConfig) -> None:
        super().__init__(global_config)
        if not self.llm_name.startswith(self.prefix) or len(self.llm_name) == len(self.prefix):
            raise ValueError(f"Atlas Cloud model names must use {self.prefix}<vendor/model>.")
        self.llm_base_url = validate_openai_base_url(
            self.global_config.llm_base_url or DEFAULT_ATLASCLOUD_BASE_URL,
            "chat/completions",
            "llm_base_url",
        )
        api_key = os.getenv("ATLASCLOUD_API_KEY")
        if not api_key:
            raise ValueError("ATLASCLOUD_API_KEY is required for Atlas Cloud.")
        self.cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file_name = os.path.join(self.cache_dir, f"{self.llm_name.replace('/', '_')}_cache.sqlite")
        self.max_retries = global_config.max_retry_attempts
        self._init_llm_config()
        self.openai_client = OpenAI(
            base_url=self.llm_base_url,
            api_key=api_key,
            max_retries=self.max_retries,
            timeout=5 * 60,
        )

    def _init_llm_config(self) -> None:
        generate_params = {
            "model": self.llm_name[len(self.prefix):],
        }
        if self.global_config.max_new_tokens is not None:
            generate_params["max_completion_tokens"] = self.global_config.max_new_tokens
        if self.global_config.seed is not None:
            generate_params["seed"] = self.global_config.seed
        if self.global_config.temperature is not None:
            generate_params["temperature"] = self.global_config.temperature
        self.llm_config = LLMConfig.from_dict({
            "llm_name": self.llm_name,
            "llm_base_url": self.llm_base_url,
            "generate_params": generate_params,
        })

    @cache_response
    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[str, dict, bool]:
        params = normalize_generation_token_params(self.llm_config.generate_params, kwargs, "max_completion_tokens")
        params["messages"] = messages
        logger.debug(f"Calling Atlas Cloud Chat Completions API with model {params['model']}")
        response = self.openai_client.chat.completions.create(**params)
        if len(response.choices) != 1:
            raise ValueError(f"HippoRAG expected exactly one Atlas Cloud choice, received {len(response.choices)}.")
        choice = response.choices[0]
        response_message = choice.message.content
        if not isinstance(response_message, str) or not response_message:
            refusal = getattr(choice.message, "refusal", None)
            detail = f" Refusal: {refusal}" if refusal else ""
            raise ValueError(f"Atlas Cloud response did not contain non-empty text.{detail}")
        usage = response.usage
        if usage is None:
            raise ValueError("Atlas Cloud response omitted usage; HippoRAG cannot account for this request safely.")
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
