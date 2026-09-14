import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hipporag.llm import _get_llm_class
from hipporag.llm.atlascloud_llm import AtlasCloudLLM
from hipporag.utils.config_utils import BaseConfig


class AtlasCloudLLMTest(unittest.TestCase):
    def make_config(self, save_dir):
        return BaseConfig(
            llm_name="atlascloud/deepseek-ai/deepseek-v3.2",
            save_dir=save_dir,
        )

    def test_provider_selection(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "apikey-test"}):
            with patch("hipporag.llm.atlascloud_llm.OpenAI"):
                self.assertIsInstance(_get_llm_class(self.make_config(save_dir)), AtlasCloudLLM)

    def test_chat_completions_inference(self):
        response = SimpleNamespace(
            id="chatcmpl-atlas-test",
            model="atlascloud/deepseek-ai/deepseek-v3.2",
            choices=[SimpleNamespace(message=SimpleNamespace(content="HippoRAG Atlas Cloud test passed"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=5, total_tokens=9),
        )
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "apikey-test"}):
            with patch("hipporag.llm.atlascloud_llm.OpenAI") as openai:
                openai.return_value.chat.completions.create = MagicMock(return_value=response)
                llm = AtlasCloudLLM(self.make_config(save_dir))
                message, metadata, cached = llm.infer([{"role": "user", "content": "Test"}])

        self.assertEqual(message, "HippoRAG Atlas Cloud test passed")
        self.assertEqual(metadata["prompt_tokens"], 4)
        self.assertEqual(metadata["total_tokens"], 9)
        self.assertFalse(cached)
        openai.return_value.chat.completions.create.assert_called_once_with(
            model="deepseek-ai/deepseek-v3.2",
            max_completion_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": "Test"}],
        )

    def test_missing_api_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ATLASCLOUD_API_KEY"):
                AtlasCloudLLM(self.make_config(save_dir))

    def test_bare_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "apikey-test"}):
            config = self.make_config(save_dir)
            config.llm_name = "atlascloud/"
            with self.assertRaisesRegex(ValueError, "vendor/model"):
                AtlasCloudLLM(config)

    def test_default_base_url_is_used_when_unset(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "apikey-test"}):
            with patch("hipporag.llm.atlascloud_llm.OpenAI") as openai:
                llm = AtlasCloudLLM(self.make_config(save_dir))

        self.assertEqual(llm.llm_base_url, "https://api.atlascloud.ai/v1")
        self.assertEqual(openai.call_args.kwargs["base_url"], "https://api.atlascloud.ai/v1")


if __name__ == "__main__":
    unittest.main()
