import unittest
from hipporag.information_extraction.openie_openai import _extract_json_list_field, _extract_ner_from_response
from hipporag.embedding_model import _get_embedding_model_class, OpenAIEmbeddingModel


class TestOpenIEExtraction(unittest.TestCase):
    def test_standard_json_object(self):
        resp = '{"named_entities": ["Apple", "California"]}'
        self.assertEqual(_extract_ner_from_response(resp), ["Apple", "California"])

    def test_markdown_code_fence(self):
        resp = "```json\n{\n  \"named_entities\": [\"Tesla\", \"Austin\"]\n}\n```"
        self.assertEqual(_extract_ner_from_response(resp), ["Tesla", "Austin"])

    def test_bare_json_list(self):
        self.assertEqual(_extract_ner_from_response('\n\n["Oliver Badman", "Montebello"]'), ["Oliver Badman", "Montebello"])
        self.assertEqual(_extract_ner_from_response("```json\n[\"Tesla\", \"Austin\"]\n```"), ["Tesla", "Austin"])
        self.assertEqual(_extract_ner_from_response("[]"), [])

    def test_object_format_takes_precedence_over_bare_list(self):
        self.assertEqual(_extract_ner_from_response('Entities: {"named_entities": ["Apple"]}'), ["Apple"])

    def test_neither_format_is_an_error(self):
        invalid_responses = [
            "",
            "Apple, California",
            '{"entities": ["Apple"]}',
            '{"named_entities": "Apple"}',
            '[["Apple", "located in", "California"]]',
            '["Apple", 1]',
            'The entities are ["Apple", "California"].',
            '["Apple", "Calif',
        ]
        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, "bare JSON list of entity strings"):
                    _extract_ner_from_response(response)

    def test_triple_extraction_still_requires_object(self):
        with self.assertRaisesRegex(ValueError, "'triples'"):
            _extract_json_list_field('[["Apple", "located in", "California"]]', "triples")

    def test_embedding_model_fallback(self):
        cls = _get_embedding_model_class("Nemotron-3-Embed-1B-NVFP4")
        self.assertEqual(cls, OpenAIEmbeddingModel)
        cls2 = _get_embedding_model_class("jina-embeddings-v3")
        self.assertEqual(cls2, OpenAIEmbeddingModel)


if __name__ == "__main__":
    unittest.main()
