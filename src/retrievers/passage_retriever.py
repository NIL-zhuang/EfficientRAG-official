import argparse
import os
import sys
from glob import glob
from typing import List, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from retrievers.embeddings import (
    Embedder,
    EmbeddingModelTypes,
    ModelCheckpointMapping,
    ModelTypes,
)
from retrievers.utils.utils import load_passages
from retrievers.vector_index import FaissIndex
from retrievers.vector_index.faiss_index import IndexType

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Retriever(object):
    def __init__(
        self,
        passage_path: str,
        passage_embedding_path: str = None,
        index_path_dir: str = None,
        model_type: EmbeddingModelTypes = "e5-base-v2",
        model_path: str = None,
        save_or_load_index: bool = True,
        batch_size: int = 128,
        embed_vector_dim: int = None,
        index_type: IndexType = "Flat",
        max_search_batch_size: int = 2048,
    ):
        if model_path is None:
            model_path = ModelCheckpointMapping[model_type]
        self.embedder = Embedder(model_type, model_path, batch_size)
        if embed_vector_dim is None:
            embed_vector_dim = self.embedder.get_dim()
        self.index = FaissIndex(embed_vector_dim, index_type, max_search_batch_size)
        if index_path_dir is None:
            index_path_dir = passage_embedding_path

        if save_or_load_index and self.index.exist_index(index_path_dir):
            print(f"Loading index from {index_path_dir}")
            self.index.deserialize(index_path_dir)
        else:
            print(f"Building index from {passage_embedding_path}")
            self.load_embeddings(passage_embedding_path)
            if save_or_load_index:
                print(f"Saving index to {index_path_dir}")
                self.index.serialize(index_path_dir)
        print(f"Loading passages from {passage_path}")
        passages = load_passages(passage_path)
        self.passage_map = {p["id"]: p for p in passages}
        print(f"Loaded {len(passages)} passages.")

    def load_embeddings(self, passage_embedding_path):
        embedding_file = sorted(glob(f"{passage_embedding_path}/passage*"))
        self.index.load_data(embedding_file)

    def search(self, query: Union[str, List[str]], top_k: int = 10):
        query = [query] if isinstance(query, str) else query
        query_vectors = self.embedder.embed(query)
        top_ids_scores = self.index.search(query_vectors, top_k)
        # convert passage id to passage
        docs = [
            [self.passage_map[doc_id] for doc_id in top_docs]
            for top_docs, top_scores in top_ids_scores
        ]
        docs = [doc_list[:top_k] for doc_list in docs]
        return docs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--passages", type=str, required=True, help="document file path"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="e5-base-v2",
        choices=list(ModelTypes.keys()),
        help="Embedding Model",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Embedding model checkpoint",
    )
    parser.add_argument("--save_or_load_index", action="store_true")
    parser.add_argument(
        "--embeddings", type=str, required=True, help="Document embedding path"
    )
    parser.add_argument("--query", type=str, help="query")
    args = parser.parse_args()
    return args


def test(opt: argparse.Namespace):
    retriever = Retriever(
        opt.passages,
        opt.embeddings,
        model_type=opt.model_type,
        model_path=opt.model_name_or_path,
    )
    if opt.query is None:
        queries = [
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            "What is the difference between llama and alpaca?",
        ]
    else:
        queries = [opt.query]
    docs = retriever.search(queries, 20)
    print(docs)


if __name__ == "__main__":
    options = parse_args()
    test(options)
