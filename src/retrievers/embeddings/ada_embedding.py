import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from language_models import AOAI

from .dense_embedding import DenseEmbedding


class AdaEmbedding(DenseEmbedding):
    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        api_version: str = "2024-02-15-preview",
    ):
        super().__init__(embedding_model, embedding_vector_size=1536)
        self.model = AOAI(embedding_model=embedding_model, api_version=api_version)

    def instantiate(self):
        return

    def embed(self, query: str):
        return self.model.embed(query)

    def embed_batch(self, queries: list[str], max_workers: int = 50):
        return self.model.embed(queries)
