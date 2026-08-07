import requests
import torch


def call_generate(
    input_txt: str,
    context_txt: str,
    window_size: int | None = None,
    max_new_tokens: int | None = None,
    host: str = "http://127.0.0.1:8989",
    timeout: int = 120,
) -> torch.Tensor:
    """Send the prompt to the API server and return the generated token tensor."""
    payload: dict[str, object] = {
        "input_txt": input_txt,
        "context_txt": context_txt,
    }
    if window_size is not None:
        payload["window_size"] = int(window_size)
    if max_new_tokens is not None:
        payload["max_new_tokens"] = int(max_new_tokens)

    response = requests.post(f"{host}/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "output" not in data:
        raise ValueError(f"Unexpected response payload: {data}")
    return torch.tensor([data["output"]])


def check_server_health(host: str = "http://127.0.0.1:8989", timeout: int = 60) -> None:
    """Check if the API server is healthy and responding."""
    try:
        response = requests.get(f"{host}/health", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"Server is not healthy: {data}")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to server at {host}. Is the server running?"
        ) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"Server health check timed out after {timeout}s") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Server health check failed: {e}") from e
    print("Server is healthy.")


class GenerativeAdapter(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.base_model = model  # placeholder
        self.tokenizer = tokenizer
        check_server_health()

    @property
    def generation_config(self):
        return self.base_model.generation_config

    def generate(self, *args, **kwargs):
        ctx_ids = kwargs["ctx_ids"]
        input_ids = kwargs["input_ids"]
        assert ctx_ids.shape[0] == 1
        assert input_ids.shape[0] == 1

        context_txt = self.tokenizer.decode(ctx_ids[0])
        input_txt = self.tokenizer.decode(input_ids[0])
        outputs = call_generate(input_txt, context_txt)
        return outputs
