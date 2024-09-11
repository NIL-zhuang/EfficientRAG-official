from typing import Literal, Union

import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbedding

Pooling = Union[str, Literal["average", "cls"]]


class DenseEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name_or_path: str,
        embedding_vector_size: int,
        no_fp16: bool = False,
        pooling_type: Pooling = "average",
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.embedding_vector_size = embedding_vector_size
        self.model = None
        self.tokenizer = None
        self.fp16 = not no_fp16
        self.pooling_type = pooling_type

    def instantiate(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.model.eval()
        self.model = self.model.cuda()
        if self.fp16:
            self.model = self.model.half()

    def embed(self, query: str):
        if self.model is None:
            self.instantiate()
        queries = [query]
        return self.embed_batch(queries)[0]

    def embed_batch(self, queries: list[str]):
        if self.model is None:
            self.instantiate()

        with torch.no_grad():
            inputs = self.tokenizer(
                queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = self.pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = embeddings.cpu().numpy()
            return embeddings

    def pooling(self, last_hidden_states, attention_mask) -> torch.Tensor:
        if self.pooling_type == "average":
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling_type == "cls":
            return last_hidden_states[:, 0, :]
        else:
            raise NotImplementedError
