import json
from typing import Dict, Tuple

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo
from ..utils.misc_utils import NerRawOutput, TripleRawOutput
from ..utils.logging_utils import get_logger
from ..prompts import PromptTemplateManager
from ..llm.vllm_offline import VLLMOffline

logger = get_logger(__name__)


class VLLMOfflineOpenIE(OpenIE):
    def __init__(self, global_config):

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = VLLMOffline(global_config)

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using vLLM offline batch mode, including NER and triple extraction

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

        ner_input_messages = [self.prompt_template_manager.render(name='ner', passage=p) for p in chunk_passages.values()]
        ner_output, ner_output_metadata = self.llm_model.batch_infer(ner_input_messages, json_template='ner', max_tokens=512)

        triple_extract_input_messages = [self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=named_entities
        ) for passage, named_entities in zip(chunk_passages.values(), ner_output)]
        triple_output, triple_output_metadata = self.llm_model.batch_infer(triple_extract_input_messages, json_template='triples', max_tokens=2048)

        ner_raw_outputs = []
        for idx, ner_output_instance in enumerate(ner_output):
            chunk_id = list(chunks.keys())[idx]
            response = ner_output_instance
            try:
                unique_entities = json.loads(response)["named_entities"]
            except Exception as e:
                unique_entities = []
                logger.warning(f"Could not parse response from OpenIE: {e}")
            if len(unique_entities) == 0:
                logger.warning("No entities extracted for chunk_id: {}".format(chunk_id))
            ner_raw_output = NerRawOutput(chunk_id, response, unique_entities, {})
            ner_raw_outputs.append(ner_raw_output)
        ner_results_dict = {chunk_key: ner_raw_output for chunk_key, ner_raw_output in zip(chunks.keys(), ner_raw_outputs)}

        triple_raw_outputs = []
        for idx, triple_output_instance in enumerate(triple_output):
            chunk_id = list(chunks.keys())[idx]
            response = triple_output_instance
            try:
                triples = json.loads(response)["triples"]
            except:
                triples = []
                logger.warning(f"Could not parse response from OpenIE: {e}")
            if len(triples) == 0:
                logger.warning("No triples extracted for chunk_id: {}".format(chunk_id))
            triple_raw_output = TripleRawOutput(chunk_id, response, triples, {})
            triple_raw_outputs.append(triple_raw_output)
        triple_results_dict = {chunk_key: triple_raw_output for chunk_key, triple_raw_output in zip(chunks.keys(), triple_raw_outputs)}

        return ner_results_dict, triple_results_dict
