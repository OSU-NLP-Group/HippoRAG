from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM
from .bedrock_llm import BedrockLLM
from .bedrock_mantle import BedrockMantleLLM
from .atlascloud_llm import AtlasCloudLLM
from .orcarouter_llm import OrcaRouterLLM
from .transformers_llm import TransformersLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    if config.llm_name.startswith('atlascloud/'):
        return AtlasCloudLLM(config)

    if config.llm_name.startswith('orcarouter/'):
        return OrcaRouterLLM(config)

    if config.llm_name.startswith('bedrock-mantle/'):
        return BedrockMantleLLM(config)

    if config.llm_name.startswith('bedrock/'):
        return BedrockLLM(config)

    if config.llm_name.startswith('Transformers/'):
        return TransformersLLM(config)

    return CacheOpenAI.from_experiment_config(config)
