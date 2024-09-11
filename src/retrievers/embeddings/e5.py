import torch
from torch._tensor import Tensor
import torch.nn.functional as F

from .dense_embedding import DenseEmbedding


class E5Embedding(DenseEmbedding):
    def __init__(
        self,
        model_name_or_path: str,
        embedding_vector_size: int,
        pooling_type: str = None,
    ):
        if pooling_type is None:
            pooling_type = "e5-average"
        super().__init__(
            model_name_or_path=model_name_or_path,
            embedding_vector_size=embedding_vector_size,
            pooling_type=pooling_type,
        )

    def pooling(self, last_hidden_states, attention_mask) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class E5BaseV2Embedding(E5Embedding):
    def __init__(self, model_name_or_path: str = None):
        if model_name_or_path is None:
            model_name_or_path = "intfloat/e5-base-v2"
        super().__init__(
            model_name_or_path=model_name_or_path,
            embedding_vector_size=768,
        )


class E5LargeV2Embedding(E5Embedding):
    def __init__(self, model_name_or_path: str = None):
        if model_name_or_path is None:
            model_name_or_path = "intfloat/e5-large-v2"
        super().__init__(
            model_name_or_path=model_name_or_path,
            embedding_vector_size=1024,
        )


class E5MistralInstructEmbedding(E5Embedding):
    def __init__(self, model_name_or_path: str = None):
        if model_name_or_path is None:
            model_name_or_path = "intfloat/e5-mistral-7b-instruct"
        super().__init__(
            model_name_or_path=model_name_or_path,
            embedding_vector_size=4096,
            pooling_type="last_token_pool",
        )

        self.template = "Instruct: {task_description}\nQuery: {query}"
        self.max_length = 4096

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return self.template.format(task_description=task_description, query=query)

    def pooling(self, last_hidden_states, attention_mask) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            embeddings = last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            embeddings = last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def embed_batch(self, queries: list[str]):
        if self.model is None or self.tokenizer is None:
            self.instantiate()

        with torch.no_grad():
            batch_dict = self.tokenizer(
                queries,
                max_length=self.max_length - 1,
                return_attention_mask=False,
                padding=False,
                truncation=True,
            )
            batch_dict["input_ids"] = [
                input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]
            ]
            batch_dict = self.tokenizer.pad(
                batch_dict,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            batch_dict = {k: v.cuda() for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = self.pooling(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = embeddings.cpu().numpy()
        return embeddings
