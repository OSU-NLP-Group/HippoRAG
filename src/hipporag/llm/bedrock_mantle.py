import os
from copy import deepcopy
from typing import List, Tuple

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from openai import OpenAI

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig
from .openai_gpt import cache_response

logger = get_logger(__name__)


class BedrockMantleSigV4Auth(httpx.Auth):
    requires_request_body = True

    def __init__(self, profile_name: str, region_name: str) -> None:
        self.session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self.region_name = region_name

    def auth_flow(self, request):
        credentials = self.session.get_credentials()
        if credentials is None:
            raise ValueError("No AWS credentials were found for Bedrock Mantle SigV4 authentication.")
        aws_request = AWSRequest(method=request.method, url=str(request.url), data=request.content, headers=dict(request.headers))
        SigV4Auth(credentials.get_frozen_credentials(), "bedrock-mantle", self.region_name).add_auth(aws_request)
        request.headers.update(dict(aws_request.headers.items()))
        yield request


class BedrockMantleLLM(BaseLLM):
    """Amazon Bedrock Mantle implementation using the OpenAI Responses API."""

    prefix = "bedrock-mantle/"

    def __init__(self, global_config: BaseConfig) -> None:
        super().__init__(global_config)
        if not self.llm_name.startswith(self.prefix) or len(self.llm_name) == len(self.prefix):
            raise ValueError(f"Bedrock Mantle model names must use {self.prefix}<model-id>.")
        if not self.global_config.llm_base_url:
            raise ValueError("llm_base_url is required for Bedrock Mantle, for example https://bedrock-mantle.us-east-2.api.aws/openai/v1.")
        self.cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file_name = os.path.join(self.cache_dir, f"{self.llm_name.replace('/', '_')}_cache.sqlite")
        self.max_retries = global_config.max_retry_attempts
        self._init_llm_config()
        if self.global_config.bedrock_mantle_auth == "api_key":
            api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            if not api_key:
                raise ValueError("AWS_BEARER_TOKEN_BEDROCK is required when bedrock_mantle_auth is api_key.")
            self.openai_client = OpenAI(base_url=self.global_config.llm_base_url, api_key=api_key, max_retries=self.max_retries)
        elif self.global_config.bedrock_mantle_auth == "aws_credentials":
            if not self.global_config.bedrock_region:
                raise ValueError("bedrock_region is required when bedrock_mantle_auth is aws_credentials.")
            auth = BedrockMantleSigV4Auth(self.global_config.bedrock_aws_profile, self.global_config.bedrock_region)
            self.openai_client = OpenAI(base_url=self.global_config.llm_base_url, api_key="bedrock-sigv4", http_client=httpx.Client(auth=auth), max_retries=self.max_retries)
        else:
            raise ValueError(f"Unsupported Bedrock Mantle authentication method: {self.global_config.bedrock_mantle_auth}")

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig.from_dict({
            **self.global_config.__dict__,
            "generate_params": {
                "model": self.llm_name[len(self.prefix):],
                "max_output_tokens": self.global_config.max_new_tokens,
                "store": False,
            },
        })

    @cache_response
    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[str, dict]:
        params = deepcopy(self.llm_config.generate_params)
        params.update(kwargs)
        params["input"] = messages
        logger.debug(f"Calling Amazon Bedrock Mantle Responses API with model {params['model']}")
        response = self.openai_client.responses.create(**params)
        message = response.output_text
        if not isinstance(message, str):
            raise TypeError("Bedrock Mantle response.output_text must be a string.")
        usage = response.usage
        metadata = {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "response_id": response.id,
            "status": response.status,
        }
        return message, metadata
