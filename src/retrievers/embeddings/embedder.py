import warnings
from typing import Literal

import numpy as np
import torch
from tqdm import TqdmExperimentalWarning, tqdm
from tqdm.rich import tqdm_rich

from .ada_embedding import AdaEmbedding
from .contriever import Contriever
from .e5 import E5BaseV2Embedding, E5LargeV2Embedding
from .utils.normalize_text import normalize

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


EmbeddingModelTypes = Literal[
    "contriever",
    "e5-base-v2",
    "e5-large-v2",
    "e5-mistral-instruct",
    "ada-002",
]

ModelTypes = {
    "contriever": Contriever,
    "e5-base-v2": E5BaseV2Embedding,
    "e5-large-v2": E5LargeV2Embedding,
    "ada-002": AdaEmbedding,
}

ModelCheckpointMapping = {
    "contriever": "model_cache/contriever-msmarco",
    "e5-base-v2": "model_cache/e5-base-v2",
    "e5-large-v2": "model_cache/e5-large-v2",
    "ada-002": "text-embedding-ada-002",
}


class Embedder(object):
    def __init__(
        self,
        model_type: EmbeddingModelTypes,
        model_name_or_path: str = None,
        batch_size: int = 128,
        chunk_size: int = int(2e6),
        text_lower_case: bool = False,
        text_normalize: bool = False,
        no_title: bool = False,
    ):
        if model_name_or_path is None:
            model_name_or_path = ModelCheckpointMapping[model_type]
        self.embedder = ModelTypes[model_type](model_name_or_path)
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        self.text_lower_case = text_lower_case
        self.text_normalize = text_normalize
        self.no_title = no_title

    def process_text(self, line):
        if isinstance(line, dict):
            if self.no_title or "title" not in line:
                text = line["text"]
            else:
                text = f"{line['title']}: {line['text']}"
        else:
            text = line

        if self.text_lower_case:
            text = text.lower()
        if self.text_normalize:
            text = normalize(text)
        return text

    def get_ids(self, data):
        return [line["id"] for line in data]

    def embed_passages(self, data):
        ids = self.get_ids(data)
        texts = [self.process_text(line) for line in data]

        chunkBatch = (len(texts) - 1) // self.chunk_size + 1
        with torch.no_grad():
            for idx in range(chunkBatch):
                print(f"Processing chunk {idx + 1}/{chunkBatch}")
                chunkStartIdx = idx * self.chunk_size
                chunkEndIdx = min((idx + 1) * self.chunk_size, len(texts))
                chunk = texts[chunkStartIdx:chunkEndIdx]
                chunk_ids = ids[chunkStartIdx:chunkEndIdx]
                chunk_embeddings = self.embed(chunk, verbose=True)
                yield idx, (chunk_ids, chunk_embeddings)

    def embed(self, textBatch, verbose=False):
        embeddings = np.array([])
        textBatch = [self.process_text(text) for text in textBatch]
        batches = (len(textBatch) - 1) // self.batch_size + 1
        with torch.no_grad():
            if verbose:
                iter_range = tqdm_rich(range(batches), desc="Embedding")
            else:
                iter_range = range(batches)
            # for idx in tqdm_rich(range(batches), desc="Embedding"):
            for idx in iter_range:
                start_idx = idx * self.batch_size
                end_idx = min((idx + 1) * self.batch_size, len(textBatch))
                batch = textBatch[start_idx:end_idx]
                curEmbeddings = self.embedder.embed_batch(batch)
                embeddings = np.vstack((embeddings, curEmbeddings)) if embeddings.size else curEmbeddings
        return embeddings

    def get_dim(self):
        return self.embedder.embedding_vector_size
