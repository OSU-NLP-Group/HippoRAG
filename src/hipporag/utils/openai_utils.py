import ipaddress
import os
import warnings
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit


@dataclass(frozen=True)
class AzureOpenAISettings:
    endpoint: str
    api_version: str
    deployment: Optional[str]


def _validate_url(url: str, setting_name: str):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{setting_name} must be an absolute HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{setting_name} must not contain credentials.")
    if parsed.fragment:
        raise ValueError(f"{setting_name} must not contain a URL fragment.")
    return parsed


def validate_openai_base_url(base_url: Optional[str], operation_path: str, setting_name: str) -> Optional[str]:
    if base_url is None:
        return None
    parsed = _validate_url(base_url, setting_name)
    if parsed.query:
        raise ValueError(f"{setting_name} must not contain query parameters.")
    normalized_path = parsed.path.rstrip("/").lower()
    terminal_path = "/" + operation_path.strip("/").lower()
    if normalized_path.endswith(terminal_path):
        raise ValueError(
            f"{setting_name} must be an API base URL, not a full {terminal_path} operation URL; "
            "the OpenAI client appends the operation path automatically."
        )
    return base_url.rstrip("/")


def resolve_openai_api_key(base_url: Optional[str], api_key_env: Optional[str], setting_name: str) -> Optional[str]:
    """Read the key from ``api_key_env`` when configured; otherwise defer to the OpenAI SDK default."""
    if api_key_env is None:
        return local_openai_api_key(base_url)
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is required because {setting_name}={api_key_env!r} is set.")
    return api_key


def local_openai_api_key(base_url: Optional[str]) -> Optional[str]:
    if base_url is None or os.getenv("OPENAI_API_KEY"):
        return None
    hostname = urlsplit(base_url).hostname
    if hostname is None:
        return None
    if hostname.lower() == "localhost":
        return "local-placeholder"
    try:
        if ipaddress.ip_address(hostname).is_loopback:
            return "local-placeholder"
    except ValueError:
        return None
    return None


def resolve_azure_openai_settings(
    endpoint: str,
    *,
    api_version: Optional[str],
    deployment: Optional[str],
    operation: str,
) -> AzureOpenAISettings:
    if operation not in {"chat.completions", "embeddings"}:
        raise ValueError(f"Unsupported Azure OpenAI operation: {operation}.")
    parsed = _validate_url(endpoint, "Azure OpenAI endpoint")
    query = parse_qs(parsed.query, keep_blank_values=True)
    unknown_query = sorted(set(query) - {"api-version"})
    if unknown_query:
        raise ValueError(f"Azure OpenAI endpoint has unsupported query parameters: {', '.join(unknown_query)}.")
    query_versions = query.get("api-version", [])
    if len(set(query_versions)) > 1:
        raise ValueError("Azure OpenAI endpoint contains conflicting api-version values.")
    query_version = query_versions[0] if query_versions else None
    if query_version == "":
        raise ValueError("Azure OpenAI api-version cannot be empty.")
    if api_version and query_version and api_version != query_version:
        raise ValueError("Azure OpenAI api_version conflicts with the api-version in the endpoint URL.")

    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    deployment_index = next(
        (index for index in range(len(path_parts) - 1) if path_parts[index:index + 2] == ["openai", "deployments"]),
        None,
    )
    legacy_deployment = None
    is_legacy_operation_url = deployment_index is not None
    if is_legacy_operation_url:
        if len(path_parts) <= deployment_index + 2:
            raise ValueError("Azure OpenAI operation URL is missing its deployment name.")
        legacy_deployment = path_parts[deployment_index + 2]
        suffix = path_parts[deployment_index + 3:]
        expected_suffix = {"chat.completions": ["chat", "completions"], "embeddings": ["embeddings"]}[operation]
        if suffix != expected_suffix:
            raise ValueError(f"Azure OpenAI endpoint is not a {operation} operation URL.")
        base_path_parts = path_parts[:deployment_index]
        base_path = "/" + "/".join(base_path_parts) if base_path_parts else ""
    else:
        forbidden_suffix = ["chat", "completions"] if operation == "chat.completions" else ["embeddings"]
        if path_parts[-len(forbidden_suffix):] == forbidden_suffix:
            raise ValueError("Azure OpenAI full operation URLs must include /openai/deployments/<deployment>/.")
        base_path = parsed.path.rstrip("/")

    if deployment and legacy_deployment and deployment != legacy_deployment:
        raise ValueError("Azure OpenAI deployment conflicts with the deployment in the endpoint URL.")
    resolved_deployment = deployment or legacy_deployment
    if resolved_deployment is not None and (not isinstance(resolved_deployment, str) or not resolved_deployment.strip() or "/" in resolved_deployment):
        raise ValueError("Azure OpenAI deployment must be a non-empty deployment name without slashes.")
    resolved_api_version = api_version or query_version
    if not isinstance(resolved_api_version, str) or not resolved_api_version.strip():
        raise ValueError("Azure OpenAI api_version is required; set it explicitly or include ?api-version= in a legacy operation URL.")

    if is_legacy_operation_url or query_version:
        warnings.warn(
            "Passing api-version or a full operation URL inside the Azure endpoint is deprecated; "
            "configure the resource endpoint, api_version, and deployment separately.",
            FutureWarning,
            stacklevel=2,
        )
    clean_endpoint = urlunsplit((parsed.scheme, parsed.netloc, base_path, "", "")).rstrip("/")
    return AzureOpenAISettings(clean_endpoint, resolved_api_version, resolved_deployment)
