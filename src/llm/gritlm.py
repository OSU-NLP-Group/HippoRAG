# See https://github.com/ContextualAI/gritlm
from gritlm import GritLM

from src.llm import EmbeddingModelWrapper


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


class GritWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str = 'GritLM/GritLM-7B', **kwargs):
        """
        Loads the model for both capabilities; If you only need embedding pass `mode="embedding"` to save memory (no lm head).
        To load the 8x7B you will likely need multiple GPUs.
        @param model_name:
        @param kwargs:
        """
        self.model = GritLM(model_name, torch_dtype='auto', **kwargs)

    def encode_list(self, texts: list, instruction: str):
        return self.model.encode(texts, instruction=gritlm_instruction(instruction))

    def encode_text(self, text, instruction: str = '', norm=True, return_numpy=False, return_cpu=False):
        if isinstance(text, str):
            text = [text]
        if isinstance(text, list):
            res = self.encode_list(text, instruction)
        else:
            raise ValueError(f"Expected str or list, got {type(text)}")
        if return_cpu:
            res = res.to('cpu')
        if return_numpy:
            res = res.numpy()

    def cosine_sim(self, query_rep, doc_rep):
        from scipy.spatial.distance import cosine
        return 1 - cosine(query_rep, doc_rep)

    def generate(self, messages: list, max_new_tokens=256, do_sample=False):
        """

        @param messages: a list, e.g., [{"role": "user", "content": "Please write me a poem."}]
        @return:
        """
        encoded = self.model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        encoded = encoded.to(self.model.device)
        gen = self.model.generate(encoded, max_new_tokens=max_new_tokens, do_sample=do_sample)
        decoded = self.model.tokenizer.batch_decode(gen)
        return decoded
