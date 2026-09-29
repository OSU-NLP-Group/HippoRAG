import json
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_json_list_field(response: str, field_name: str) -> List:
    decoder = json.JSONDecoder()
    for start_index, character in enumerate(response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response[start_index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or field_name not in payload:
            continue
        value = payload[field_name]
        if not isinstance(value, list):
            raise ValueError(f"OpenIE response field {field_name!r} must be a list.")
        return value
    raise ValueError(f"OpenIE response does not contain a valid JSON object with {field_name!r}.")


def _parse_bare_json_string_list(response: str):
    """Return the response as a list of strings if it is exactly one JSON array of strings, else None."""
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        return None
    return payload


def _extract_ner_from_response(real_response):
    # The one-shot example is an object so the prompt also works with OpenAI JSON mode, but the
    # system prompt asks for "a JSON list", which newer open models follow literally.
    try:
        return _extract_json_list_field(real_response, "named_entities")
    except ValueError as object_error:
        entities = _parse_bare_json_string_list(real_response)
        if entities is None:
            raise ValueError(
                f"{object_error} Expected {{\"named_entities\": [...]}} or a bare JSON list of entity strings."
            ) from None
        return entities


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, max_workers: int = 8, ner_max_tokens: int = 2048, triple_max_tokens: int = 4096):
        # Init prompt template manager
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if ner_max_tokens < 1 or triple_max_tokens < 1:
            raise ValueError("OpenIE token limits must be at least 1.")
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.max_workers = max_workers
        self.ner_max_tokens = ner_max_tokens
        self.triple_max_tokens = triple_max_tokens

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            inference_kwargs = {"max_new_tokens": self.ner_max_tokens}
            response_format = getattr(getattr(self.llm_model, "global_config", None), "response_format", None)
            if response_format is not None:
                inference_kwargs["response_format"] = response_format
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
                **inference_kwargs,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            return _extract_json_list_field(real_response, "triples")

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            inference_kwargs = {"max_new_tokens": self.triple_max_tokens}
            response_format = getattr(getattr(self.llm_model, "global_config", None), "response_format", None)
            if response_format is not None:
                inference_kwargs["response_format"] = response_format
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
                **inference_kwargs,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        logical_prompt_tokens = 0
        logical_completion_tokens = 0
        billable_prompt_tokens = 0
        billable_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                prompt_tokens = metadata.get('prompt_tokens', 0)
                completion_tokens = metadata.get('completion_tokens', 0)
                logical_prompt_tokens += prompt_tokens
                logical_completion_tokens += completion_tokens
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                else:
                    billable_prompt_tokens += prompt_tokens
                    billable_completion_tokens += completion_tokens

                pbar.set_postfix({
                    'billable_prompt_tokens': billable_prompt_tokens,
                    'billable_completion_tokens': billable_completion_tokens,
                    'logical_prompt_tokens': logical_prompt_tokens,
                    'logical_completion_tokens': logical_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        failed_ner_chunk_ids = [result.chunk_id for result in ner_results_list if result.metadata.get("error")]
        if failed_ner_chunk_ids:
            raise RuntimeError(f"NER failed for {len(failed_ner_chunk_ids)} chunk(s): {failed_ner_chunk_ids}")

        triple_results_list = []
        logical_prompt_tokens, logical_completion_tokens = 0, 0
        billable_prompt_tokens, billable_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                prompt_tokens = metadata.get('prompt_tokens', 0)
                completion_tokens = metadata.get('completion_tokens', 0)
                logical_prompt_tokens += prompt_tokens
                logical_completion_tokens += completion_tokens
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                else:
                    billable_prompt_tokens += prompt_tokens
                    billable_completion_tokens += completion_tokens
                pbar.set_postfix({
                    'billable_prompt_tokens': billable_prompt_tokens,
                    'billable_completion_tokens': billable_completion_tokens,
                    'logical_prompt_tokens': logical_prompt_tokens,
                    'logical_completion_tokens': logical_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        failed_triple_chunk_ids = [result.chunk_id for result in triple_results_list if result.metadata.get("error")]
        if failed_triple_chunk_ids:
            raise RuntimeError(f"Triple extraction failed for {len(failed_triple_chunk_ids)} chunk(s): {failed_triple_chunk_ids}")

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
