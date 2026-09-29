from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM
from .bedrock_llm import BedrockLLM
from .bedrock_mantle import BedrockMantleLLM
from .gateway_llm import OpenAICompatibleGatewayLLM
from .gateways import GATEWAYS, GatewaySpec, find_gateway
from .transformers_llm import TransformersLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    gateway = find_gateway(config.llm_name)
    if gateway is not None:
        return OpenAICompatibleGatewayLLM(config, gateway)

    if config.llm_name.startswith('bedrock-mantle/'):
        return BedrockMantleLLM(config)

    if config.llm_name.startswith('bedrock/'):
        return BedrockLLM(config)

    if config.llm_name.startswith('Transformers/'):
        return TransformersLLM(config)

    return CacheOpenAI.from_experiment_config(config)
