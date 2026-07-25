import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from hipporag.llm import _get_llm_class
from hipporag.llm.bedrock_mantle import BedrockMantleLLM, BedrockMantleSigV4Auth
from hipporag.utils.config_utils import BaseConfig


class BedrockMantleLLMTest(unittest.TestCase):
    def make_config(self, save_dir):
        return BaseConfig(
            llm_name="bedrock-mantle/openai.gpt-5.5",
            llm_base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
            save_dir=save_dir,
        )

    def test_provider_selection(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "test-key"}):
            with patch("hipporag.llm.bedrock_mantle.OpenAI"):
                self.assertIsInstance(_get_llm_class(self.make_config(save_dir)), BedrockMantleLLM)

    def test_responses_api_inference(self):
        response = SimpleNamespace(
            output_text="HippoRAG Bedrock test passed",
            id="response-id",
            status="completed",
            usage=SimpleNamespace(input_tokens=4, output_tokens=5),
        )
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "test-key"}):
            with patch("hipporag.llm.bedrock_mantle.OpenAI") as openai:
                openai.return_value.responses.create = MagicMock(return_value=response)
                llm = BedrockMantleLLM(self.make_config(save_dir))
                message, metadata, cached = llm.infer([{"role": "user", "content": "Test"}])

        self.assertEqual(message, response.output_text)
        self.assertEqual(metadata["prompt_tokens"], 4)
        self.assertFalse(cached)
        openai.return_value.responses.create.assert_called_once_with(
            model="openai.gpt-5.5",
            max_output_tokens=2048,
            store=False,
            input=[{"role": "user", "content": "Test"}],
        )

    def test_missing_api_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "AWS_BEARER_TOKEN_BEDROCK"):
                BedrockMantleLLM(self.make_config(save_dir))

    def test_aws_credentials_auth_requires_region(self):
        with tempfile.TemporaryDirectory() as save_dir:
            config = self.make_config(save_dir)
            config.bedrock_mantle_auth = "aws_credentials"
            with self.assertRaisesRegex(ValueError, "bedrock_region"):
                BedrockMantleLLM(config)

    def test_sigv4_auth_adds_authorization_header(self):
        credentials = SimpleNamespace(get_frozen_credentials=lambda: SimpleNamespace(access_key="key", secret_key="secret", token=None))
        auth = BedrockMantleSigV4Auth.__new__(BedrockMantleSigV4Auth)
        auth.session = SimpleNamespace(get_credentials=lambda: credentials)
        auth.region_name = "us-east-2"
        request = httpx.Request("POST", "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses", json={"model": "openai.gpt-5.5"})
        list(auth.auth_flow(request))
        self.assertIn("Authorization", request.headers)


if __name__ == "__main__":
    unittest.main()
