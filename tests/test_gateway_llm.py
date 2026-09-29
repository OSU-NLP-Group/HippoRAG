import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hipporag import HippoRAG
from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from hipporag.llm import GATEWAYS, _get_llm_class
from hipporag.llm.gateway_llm import OpenAICompatibleGatewayLLM
from hipporag.llm.gateways import chat_max_tokens_key, find_gateway, find_gateway_by_base_url
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.utils.config_utils import BaseConfig


UPSTREAM_MODEL = "vendor-x/model-y"


def fake_response(content="HippoRAG gateway test passed", usage=True):
    return SimpleNamespace(
        id="chatcmpl-test",
        model=UPSTREAM_MODEL,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=5, total_tokens=9) if usage else None,
    )


def expected_token_key(spec):
    return "max_completion_tokens" if spec.supports_max_completion_tokens else "max_tokens"


class GatewayRegistryTest(unittest.TestCase):
    def test_registry_is_frozen(self):
        # New OpenAI-compatible gateways belong on the generic llm_base_url + llm_api_key_env path.
        # Only extend this list for a gateway that path cannot express; see CONTRIBUTING.md.
        self.assertEqual([spec.prefix for spec in GATEWAYS], ["orcarouter/", "atlascloud/"])

    def test_registry_prefixes_are_unique_and_well_formed(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name):
                self.assertTrue(spec.prefix.endswith("/"))
                self.assertTrue(spec.default_base_url.startswith("https://"))
                self.assertTrue(spec.api_key_env.isupper())
                for other in GATEWAYS:
                    if other is not spec:
                        self.assertFalse(other.prefix.startswith(spec.prefix))

    def test_unregistered_names_and_hosts_do_not_match(self):
        for name in ("gpt-4o-mini", "bedrock/anthropic.claude", "Transformers/Qwen/Qwen3", "Qwen/Qwen3.8-Flash-Next"):
            with self.subTest(name=name):
                self.assertIsNone(find_gateway(name))
        for url in (None, "https://api.openai.com/v1", "http://localhost:8000/v1", "https://example.com/v1"):
            with self.subTest(url=url):
                self.assertIsNone(find_gateway_by_base_url(url))

    def test_chat_max_tokens_key(self):
        cases = [
            (BaseConfig(), None, "max_completion_tokens"),
            (BaseConfig(), "https://api.openai.com/v1", "max_completion_tokens"),
            (BaseConfig(azure_endpoint="https://resource.openai.azure.com"), None, "max_completion_tokens"),
            (BaseConfig(), "http://localhost:8000/v1", "max_tokens"),
            (BaseConfig(), "https://example.com/v1", "max_tokens"),
            (BaseConfig(llm_supports_max_completion_tokens=True), "http://localhost:8000/v1", "max_completion_tokens"),
            (BaseConfig(llm_supports_max_completion_tokens=False), "https://api.openai.com/v1", "max_tokens"),
        ]
        for spec in GATEWAYS:
            cases.append((BaseConfig(), spec.default_base_url, expected_token_key(spec)))
            cases.append((BaseConfig(), spec.default_base_url.upper().replace("HTTPS", "https"), expected_token_key(spec)))
        for config, base_url, expected in cases:
            with self.subTest(base_url=base_url, override=config.llm_supports_max_completion_tokens):
                self.assertEqual(chat_max_tokens_key(config, base_url), expected)


class GatewayLLMTest(unittest.TestCase):
    """Every registered gateway must satisfy the same contract."""

    def make_config(self, spec, save_dir, **kwargs):
        return BaseConfig(llm_name=f"{spec.prefix}{UPSTREAM_MODEL}", save_dir=save_dir, **kwargs)

    def test_provider_selection(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir, \
                    patch.dict(os.environ, {spec.api_key_env: "test-key"}), patch("hipporag.llm.gateway_llm.OpenAI"):
                llm = _get_llm_class(self.make_config(spec, save_dir))
                self.assertIsInstance(llm, OpenAICompatibleGatewayLLM)
                self.assertIs(llm.spec, spec)

    def test_chat_completions_inference(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir, \
                    patch.dict(os.environ, {spec.api_key_env: "test-key"}), patch("hipporag.llm.gateway_llm.OpenAI") as openai:
                openai.return_value.chat.completions.create = MagicMock(return_value=fake_response())
                llm = OpenAICompatibleGatewayLLM(self.make_config(spec, save_dir))
                message, metadata, cached = llm.infer([{"role": "user", "content": "Test"}])

                self.assertEqual(message, "HippoRAG gateway test passed")
                self.assertEqual(metadata["prompt_tokens"], 4)
                self.assertEqual(metadata["total_tokens"], 9)
                self.assertFalse(cached)
                self.assertEqual(openai.call_args.kwargs["api_key"], "test-key")
                openai.return_value.chat.completions.create.assert_called_once_with(**{
                    "model": UPSTREAM_MODEL,
                    expected_token_key(spec): 2048,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": "Test"}],
                })

    def test_explicit_token_capability_overrides_registry(self):
        for spec in GATEWAYS:
            override = not spec.supports_max_completion_tokens
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir, \
                    patch.dict(os.environ, {spec.api_key_env: "test-key"}), patch("hipporag.llm.gateway_llm.OpenAI") as openai:
                openai.return_value.chat.completions.create = MagicMock(return_value=fake_response())
                llm = OpenAICompatibleGatewayLLM(self.make_config(spec, save_dir, llm_supports_max_completion_tokens=override))
                llm.infer([{"role": "user", "content": "Test"}])
                sent = openai.return_value.chat.completions.create.call_args.kwargs
                self.assertIn("max_completion_tokens" if override else "max_tokens", sent)

    def test_missing_usage_is_an_error(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir, \
                    patch.dict(os.environ, {spec.api_key_env: "test-key"}), patch("hipporag.llm.gateway_llm.OpenAI") as openai:
                openai.return_value.chat.completions.create = MagicMock(return_value=fake_response(usage=False))
                llm = OpenAICompatibleGatewayLLM(self.make_config(spec, save_dir))
                with self.assertRaisesRegex(ValueError, f"{spec.display_name} response omitted usage"):
                    llm.infer([{"role": "user", "content": "Test"}])

    def test_missing_api_key_is_an_error(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir, \
                    patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, spec.api_key_env):
                    OpenAICompatibleGatewayLLM(self.make_config(spec, save_dir))

    def test_llm_api_key_env_overrides_registered_variable(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir, \
                    patch.dict(os.environ, {"TEAM_GATEWAY_KEY": "team-key"}, clear=True), patch("hipporag.llm.gateway_llm.OpenAI") as openai:
                OpenAICompatibleGatewayLLM(self.make_config(spec, save_dir, llm_api_key_env="TEAM_GATEWAY_KEY"))
                self.assertEqual(openai.call_args.kwargs["api_key"], "team-key")

    def test_bare_prefix_is_rejected(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir, \
                    patch.dict(os.environ, {spec.api_key_env: "test-key"}):
                config = self.make_config(spec, save_dir)
                config.llm_name = spec.prefix
                with self.assertRaisesRegex(ValueError, "vendor/model"):
                    OpenAICompatibleGatewayLLM(config)

    def test_default_base_url_is_used_when_unset(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir, \
                    patch.dict(os.environ, {spec.api_key_env: "test-key"}), patch("hipporag.llm.gateway_llm.OpenAI") as openai:
                llm = OpenAICompatibleGatewayLLM(self.make_config(spec, save_dir))
                self.assertEqual(llm.llm_base_url, spec.default_base_url)
                self.assertEqual(openai.call_args.kwargs["base_url"], spec.default_base_url)

    def test_explicit_base_url_overrides_default(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir, \
                    patch.dict(os.environ, {spec.api_key_env: "test-key"}), patch("hipporag.llm.gateway_llm.OpenAI") as openai:
                llm = OpenAICompatibleGatewayLLM(self.make_config(spec, save_dir, llm_base_url="https://proxy.invalid/v1/"))
                self.assertEqual(llm.llm_base_url, "https://proxy.invalid/v1")
                self.assertEqual(openai.call_args.kwargs["base_url"], "https://proxy.invalid/v1")

    def test_openie_provenance_records_gateway_endpoint(self):
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as save_dir:
                rag = HippoRAG.__new__(HippoRAG)
                rag.global_config = self.make_config(spec, save_dir, embedding_model_name="fake", openie_mode="online")
                rag.index_identity = "gateway-test"
                self.assertEqual(rag._current_openie_provenance()["producer"]["endpoint"], spec.default_base_url)

                rag.global_config.llm_base_url = "https://proxy.invalid/v1"
                self.assertEqual(rag._current_openie_provenance()["producer"]["endpoint"], "https://proxy.invalid/v1")


class GenericGatewayPathTest(unittest.TestCase):
    """The generic llm_base_url + llm_api_key_env path must reach any gateway exactly like its prefix does."""

    def test_generic_path_matches_named_gateway_request(self):
        messages = [{"role": "user", "content": "Test"}]
        for spec in GATEWAYS:
            with self.subTest(gateway=spec.display_name), tempfile.TemporaryDirectory() as named_dir, \
                    tempfile.TemporaryDirectory() as generic_dir, patch.dict(os.environ, {spec.api_key_env: "gateway-key"}, clear=True), \
                    patch("hipporag.llm.gateway_llm.OpenAI") as named_openai, patch("hipporag.llm.openai_gpt.OpenAI") as generic_openai:
                named_openai.return_value.chat.completions.create = MagicMock(return_value=fake_response())
                generic_openai.return_value.chat.completions.create = MagicMock(return_value=fake_response())

                named_config = BaseConfig(llm_name=f"{spec.prefix}{UPSTREAM_MODEL}", save_dir=named_dir)
                generic_config = BaseConfig(
                    llm_name=UPSTREAM_MODEL,
                    llm_base_url=spec.default_base_url,
                    llm_api_key_env=spec.api_key_env,
                    save_dir=generic_dir,
                )
                named = _get_llm_class(named_config)
                generic = _get_llm_class(generic_config)
                self.assertIsInstance(named, OpenAICompatibleGatewayLLM)
                self.assertIsInstance(generic, CacheOpenAI)

                for key in ("base_url", "api_key"):
                    self.assertEqual(named_openai.call_args.kwargs[key], generic_openai.call_args.kwargs[key])
                self.assertEqual(generic_openai.call_args.kwargs["api_key"], "gateway-key")

                named_message, named_metadata, _ = named.infer(messages)
                generic_message, generic_metadata, _ = generic.infer(messages)
                self.assertEqual(
                    named_openai.return_value.chat.completions.create.call_args.kwargs,
                    generic_openai.return_value.chat.completions.create.call_args.kwargs,
                )
                self.assertEqual((named_message, named_metadata), (generic_message, generic_metadata))

                provenance = []
                for config in (named_config, generic_config):
                    rag = HippoRAG.__new__(HippoRAG)
                    rag.global_config = config
                    rag.global_config.embedding_model_name = "fake"
                    rag.index_identity = "gateway-test"
                    provenance.append(rag._current_openie_provenance()["producer"]["endpoint"])
                self.assertEqual(provenance, [spec.default_base_url, spec.default_base_url])

    def test_llm_api_key_env_is_required_when_set(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            config = BaseConfig(llm_name=UPSTREAM_MODEL, llm_base_url="https://example.com/v1", llm_api_key_env="MISSING_KEY", save_dir=save_dir)
            with self.assertRaisesRegex(ValueError, "MISSING_KEY is required"):
                _get_llm_class(config)

    def test_unset_llm_api_key_env_keeps_openai_sdk_default(self):
        with tempfile.TemporaryDirectory() as save_dir, patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True), \
                patch("hipporag.llm.openai_gpt.OpenAI") as openai:
            _get_llm_class(BaseConfig(llm_name=UPSTREAM_MODEL, llm_base_url="https://example.com/v1", save_dir=save_dir))
            self.assertIsNone(openai.call_args.kwargs["api_key"])

    def test_gateway_llm_and_openai_embeddings_use_separate_keys(self):
        with tempfile.TemporaryDirectory() as save_dir, \
                patch.dict(os.environ, {"GATEWAY_KEY": "gateway-key", "EMBED_KEY": "embed-key"}, clear=True), \
                patch("hipporag.llm.openai_gpt.OpenAI") as llm_openai, patch("hipporag.embedding_model.OpenAI.OpenAI") as embed_openai:
            config = BaseConfig(
                llm_name=UPSTREAM_MODEL,
                llm_base_url="https://example.com/v1",
                llm_api_key_env="GATEWAY_KEY",
                embedding_model_name="text-embedding-3-small",
                embedding_api_key_env="EMBED_KEY",
                save_dir=save_dir,
            )
            _get_llm_class(config)
            OpenAIEmbeddingModel(config)
            self.assertEqual(llm_openai.call_args.kwargs["api_key"], "gateway-key")
            self.assertEqual(embed_openai.call_args.kwargs["api_key"], "embed-key")

    def test_embedding_api_key_env_is_required_when_set(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            config = BaseConfig(embedding_model_name="text-embedding-3-small", embedding_api_key_env="MISSING_EMBED_KEY")
            with self.assertRaisesRegex(ValueError, "MISSING_EMBED_KEY is required"):
                OpenAIEmbeddingModel(config)

    def test_hipporag_constructor_forwards_api_key_envs(self):
        with patch.object(HippoRAG, "_construct_or_cleanup", side_effect=RuntimeError("stop after config")):
            rag = HippoRAG.__new__(HippoRAG)
            with self.assertRaisesRegex(RuntimeError, "stop after config"):
                HippoRAG.__init__(rag, llm_api_key_env="GATEWAY_KEY", embedding_api_key_env="EMBED_KEY")
            self.assertEqual(rag.global_config.llm_api_key_env, "GATEWAY_KEY")
            self.assertEqual(rag.global_config.embedding_api_key_env, "EMBED_KEY")


if __name__ == "__main__":
    unittest.main()
