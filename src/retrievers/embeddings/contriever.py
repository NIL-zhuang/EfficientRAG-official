from .dense_embedding import DenseEmbedding


class Contriever(DenseEmbedding):
    def __init__(self, model_name_or_path: str):
        if model_name_or_path is None:
            model_name_or_path = "facebook/contriever-msmarco"
        super().__init__(
            model_name_or_path=model_name_or_path,
            embedding_vector_size=768,
            pooling_type="average",
        )
