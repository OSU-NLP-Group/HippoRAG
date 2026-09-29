from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from ..utils.config_utils import BaseConfig


@dataclass(frozen=True)
class GatewaySpec:
    """A named OpenAI-compatible gateway selected by an ``llm_name`` prefix.

    ``supports_max_completion_tokens`` records which output-cap parameter the
    gateway documents, so the request is the same whether the gateway is used
    by prefix or through ``llm_base_url``.
    """

    prefix: str
    display_name: str
    default_base_url: str
    api_key_env: str
    supports_max_completion_tokens: bool


# Frozen: any OpenAI-compatible gateway works through llm_base_url + llm_api_key_env, so new
# entries are only accepted when that generic path cannot express the gateway (see CONTRIBUTING.md).
# Entries are kept in the order they were added; registration is not an endorsement.
GATEWAYS = (
    GatewaySpec("orcarouter/", "OrcaRouter", "https://api.orcarouter.ai/v1", "ORCAROUTER_API_KEY", supports_max_completion_tokens=True),
    GatewaySpec("atlascloud/", "Atlas Cloud", "https://api.atlascloud.ai/v1", "ATLASCLOUD_API_KEY", supports_max_completion_tokens=False),
)


def find_gateway(llm_name: str) -> Optional[GatewaySpec]:
    for spec in GATEWAYS:
        if llm_name.startswith(spec.prefix):
            return spec
    return None


def find_gateway_by_base_url(base_url: Optional[str]) -> Optional[GatewaySpec]:
    hostname = urlsplit(base_url).hostname if base_url else None
    if hostname is None:
        return None
    for spec in GATEWAYS:
        if urlsplit(spec.default_base_url).hostname == hostname.lower():
            return spec
    return None


def chat_max_tokens_key(global_config: BaseConfig, base_url: Optional[str], gateway: Optional[GatewaySpec] = None) -> str:
    """Pick the Chat Completions output-cap parameter for an OpenAI-compatible endpoint.

    An explicit ``llm_supports_max_completion_tokens`` always wins. Otherwise Azure
    and official OpenAI endpoints use ``max_completion_tokens``, a registered gateway
    (by prefix or by base URL host) uses what it documents, and any other endpoint
    uses the widely supported ``max_tokens``.
    """
    supports = global_config.llm_supports_max_completion_tokens
    if supports is None:
        base_url = base_url or "https://api.openai.com/v1"
        gateway = gateway or find_gateway_by_base_url(base_url)
        if global_config.azure_endpoint is not None:
            supports = True
        elif gateway is not None:
            supports = gateway.supports_max_completion_tokens
        else:
            supports = "api.openai.com" in base_url
    return "max_completion_tokens" if supports else "max_tokens"
