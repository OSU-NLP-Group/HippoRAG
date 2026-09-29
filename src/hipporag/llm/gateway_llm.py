import os
from typing import List, Optional, Tuple

from openai import OpenAI

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from ..utils.openai_utils import validate_openai_base_url
from .base import BaseLLM, LLMConfig, normalize_generation_token_params
from .gateways import GatewaySpec, chat_max_tokens_key, find_gateway
from .openai_gpt import cache_response

logger = get_logger(__name__)


class OpenAICompatibleGatewayLLM(BaseLLM):
    """Named OpenAI-compatible gateway using the Chat Completions API.

    Model names use the ``<prefix><vendor/model>`` form; the prefix selects the
    gateway and is stripped before the request is sent. This is a shortcut for
    ``llm_name='<vendor/model>'`` with the gateway's ``llm_base_url`` and
    ``llm_api_key_env``, and sends the same request.
    """

    def __init__(self, global_config: BaseConfig, spec: Optional[GatewaySpec] = None) -> None:
        super().__init__(global_config)
        spec = spec or find_gateway(self.llm_name)
        if spec is None:
            raise ValueError(f"No OpenAI-compatible gateway is registered for model name {self.llm_name!r}.")
        self.spec = spec
        if not self.llm_name.startswith(spec.prefix) or len(self.llm_name) == len(spec.prefix):
            raise ValueError(f"{spec.display_name} model names must use {spec.prefix}<vendor/model>.")
        self.llm_base_url = validate_openai_base_url(
            self.global_config.llm_base_url or spec.default_base_url,
            "chat/completions",
            "llm_base_url",
        )
        api_key_env = self.global_config.llm_api_key_env or spec.api_key_env
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is required for {spec.display_name}.")
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
            "model": self.llm_name[len(self.spec.prefix):],
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
        name = self.spec.display_name
        target_token_key = chat_max_tokens_key(self.global_config, self.llm_base_url, self.spec)
        params = normalize_generation_token_params(self.llm_config.generate_params, kwargs, target_token_key)
        params["messages"] = messages
        logger.debug(f"Calling {name} Chat Completions API with model {params['model']}")
        response = self.openai_client.chat.completions.create(**params)
        if len(response.choices) != 1:
            raise ValueError(f"HippoRAG expected exactly one {name} choice, received {len(response.choices)}.")
        choice = response.choices[0]
        response_message = choice.message.content
        if not isinstance(response_message, str) or not response_message:
            refusal = getattr(choice.message, "refusal", None)
            detail = f" Refusal: {refusal}" if refusal else ""
            raise ValueError(f"{name} response did not contain non-empty text.{detail}")
        usage = response.usage
        if usage is None:
            raise ValueError(f"{name} response omitted usage; HippoRAG cannot account for this request safely.")
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
