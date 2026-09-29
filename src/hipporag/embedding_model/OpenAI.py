from typing import List, Optional

import numpy as np
from tqdm import tqdm
from openai import AzureOpenAI, DefaultHttpxClient, OpenAI

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from ..utils.openai_utils import resolve_azure_openai_settings, resolve_openai_api_key, validate_openai_base_url
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        self.last_usage = None
        self._instruction_warning_emitted = False
        self.request_model_name = self.embedding_model_name
        client = DefaultHttpxClient(timeout=self.global_config.embedding_request_timeout)
        try:
            if self.global_config.azure_embedding_endpoint is None:
                base_url = validate_openai_base_url(self.global_config.embedding_base_url, "embeddings", "embedding_base_url")
                self.client = OpenAI(
                    api_key=resolve_openai_api_key(base_url, self.global_config.embedding_api_key_env, "embedding_api_key_env"),
                    base_url=base_url,
                    http_client=client,
                    max_retries=self.global_config.max_retry_attempts,
                )
            else:
                settings = resolve_azure_openai_settings(
                    self.global_config.azure_embedding_endpoint,
                    api_version=self.global_config.azure_embedding_api_version or self.global_config.azure_api_version,
                    deployment=self.global_config.azure_embedding_deployment,
                    operation="embeddings",
                )
                self.request_model_name = settings.deployment or self.embedding_model_name
                self.client = AzureOpenAI(
                    api_key=resolve_openai_api_key(None, self.global_config.embedding_api_key_env, "embedding_api_key_env"),
                    api_version=settings.api_version,
                    azure_endpoint=settings.endpoint,
                    azure_deployment=settings.deployment,
                    http_client=client,
                    max_retries=self.global_config.max_retry_attempts,
                )
        except Exception:
            client.close()
            raise


    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "encode_params": {
                "batch_size": self.global_config.embedding_batch_size,
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("OpenAI embedding input cannot be empty.")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("OpenAI embedding inputs must all be strings.")
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        self.last_usage = None
        try:
            response = self.client.embeddings.create(input=texts, model=self.request_model_name, encoding_format="float")
        except Exception:
            self.last_usage = {"usage_unknown": True, "complete": False}
            raise
        usage = getattr(response, "usage", None)
        if usage is None:
            self.last_usage = {"usage_unknown": True, "complete": False}
            raise ValueError("OpenAI embedding response omitted usage; HippoRAG cannot account for this request safely.")
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if not isinstance(prompt_tokens, int) or prompt_tokens < 0 or (total_tokens is not None and (not isinstance(total_tokens, int) or total_tokens < 0)):
            self.last_usage = {"usage_unknown": True, "complete": False}
            raise ValueError("OpenAI embedding response contained invalid usage; HippoRAG cannot account for this request safely.")
        self.last_usage = {"prompt_tokens": prompt_tokens, "total_tokens": total_tokens if total_tokens is not None else prompt_tokens}
        ordered_data = sorted(response.data, key=lambda item: item.index)
        indices = [item.index for item in ordered_data]
        if indices != list(range(len(texts))):
            self.last_usage["complete"] = False
            raise ValueError(f"OpenAI embedding response indices {indices} do not match {len(texts)} inputs.")
        results = np.asarray([item.embedding for item in ordered_data], dtype=np.float32)

        return results

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str): texts = [texts]
        if not texts:
            raise ValueError("OpenAI embedding input cannot be empty.")
        unsupported = set(kwargs) - {"batch_size", "instruction", "norm"}
        if unsupported:
            raise TypeError(f"Unsupported OpenAI embedding options: {', '.join(sorted(unsupported))}.")
        # The OpenAI embeddings endpoint has no separate query/document instruction parameter.
        if kwargs.get("instruction") and not self._instruction_warning_emitted:
            logger.warning("OpenAI embeddings do not have a separate instruction parameter; HippoRAG is embedding the original text unchanged.")
            self._instruction_warning_emitted = True
        batch_size = kwargs.get("batch_size", self.embedding_config.encode_params["batch_size"])
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("OpenAI embedding batch_size must be a positive integer.")
        logger.debug(f"Calling {self.__class__.__name__} with batch_size={batch_size}")

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            results = []
            usage_totals = {"prompt_tokens": 0, "total_tokens": 0}
            try:
                with tqdm(total=len(texts), desc="Batch Encoding") as pbar:
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        results.append(self.encode(batch))
                        usage_totals["prompt_tokens"] += self.last_usage["prompt_tokens"]
                        usage_totals["total_tokens"] += self.last_usage["total_tokens"]
                        self.last_usage = {**usage_totals, "complete": False}
                        pbar.update(len(batch))
            except Exception:
                failed_usage = self.last_usage or {}
                for token_key in ("prompt_tokens", "total_tokens"):
                    token_count = failed_usage.get(token_key)
                    if isinstance(token_count, int):
                        usage_totals[token_key] += token_count
                self.last_usage = {**usage_totals, "complete": False}
                if failed_usage.get("usage_unknown") or not any(key in failed_usage for key in ("prompt_tokens", "total_tokens")):
                    self.last_usage["usage_unknown"] = True
                raise
            results = np.concatenate(results)
            self.last_usage = usage_totals
        return self._normalize_embeddings(results, normalize=kwargs.get("norm"))

    def close(self) -> None:
        self.client.close()
