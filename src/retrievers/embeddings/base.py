class BaseEmbedding(object):
    def __init__(self): ...

    def embed(self, query: str):
        raise NotImplementedError

    def embed_batch(self, queries: list[str]):
        raise NotImplementedError
