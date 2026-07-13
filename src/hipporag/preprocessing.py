from abc import ABC, abstractmethod
from typing import List, Union

from .utils.misc_utils import Chunk


class BaseTextPreprocessor(ABC):
    """Converts user documents into chunks consumed by HippoRAG."""

    @abstractmethod
    def preprocess(self, docs: List[Union[str, Chunk]]) -> List[Chunk]:
        raise NotImplementedError


class TextPreprocessor(BaseTextPreprocessor):
    """Default compatibility preprocessor that keeps one chunk per document."""

    def preprocess(self, docs: List[Union[str, Chunk]]) -> List[Chunk]:
        chunks = []
        for doc in docs:
            if isinstance(doc, Chunk):
                chunks.append(doc)
            elif isinstance(doc, str):
                chunks.append(Chunk(content=doc))
            else:
                raise TypeError(f"Documents must be strings or Chunk instances, got {type(doc).__name__}.")
        return chunks
