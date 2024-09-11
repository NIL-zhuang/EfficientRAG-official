import json
import os
from pprint import pprint
from typing import Any, Literal, NamedTuple, Union

from torch.utils.data import Dataset

from conf import DATASET_PATH

ChunkInfo = NamedTuple("ChunkInfo", id=int, title=str, chunk=str)
AllDatasets = Literal["hotpotQA", "wikiMQA", "musique"]


class MultiHopDataset(Dataset):
    """
    Base class for multi-hop datasets
    return params:
        @id: str, unique id for the question sample
        @type: str, multi-hop type of the question
            - compose: Q(explicit) -> A -> B
            - compare: Q -> A, Q -> B, (A, B)
            - inference: Q(implicit) -> A -> B
            - bridge_compare: Q -> A1 -> A2, Q -> B1 -> B2, (A2, B2)
        @question: str, the question
        @chunks: list[str], list of chunks
        @supporting_facts: list[ChunkInfo], list of supporting facts
        @decomposition:
    """

    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.data = self.load_data(data_path)

    def __getitem__(self, index):
        samples = self.data[index]
        if isinstance(samples, list):
            return [self.process(sample) for sample in samples]
        else:
            return self.process(samples)

    def process(self, sample: dict[str, Any]) -> dict[str, Any]:
        question = self.get_question(sample)
        # TBD: Whether to remove the question mark
        # if question.endswith("?"):
        #     question = question[:-1]

        chunks = self.get_chunks(sample)
        chunks = [chunk.chunk for chunk in chunks]
        answer = self.get_answer(sample)
        supporting_facts = self.get_supporting_facts(sample)
        supporting_facts = [fact._asdict() for fact in supporting_facts]

        decomposition = self.get_decomposition(sample)
        return {
            "id": self.get_id(sample),
            "type": self.get_type(sample),
            "question": question,
            "answer": answer,
            "chunks": chunks,
            "supporting_facts": supporting_facts,
            "decomposition": decomposition,
        }

    def __len__(self) -> int:
        return len(self.data)

    def get_question(self, sample: dict) -> str:
        return sample["question"]

    def get_decomposition(self, sample: dict) -> list[str]:
        return None

    def get_answer(self, sample: dict) -> str:
        return sample["answer"]

    def get_answer(self, sample: dict) -> str:
        return sample["answer"]

    def get_id(self, sample: dict) -> str:
        return sample["id"]

    def get_type(self, sample: dict) -> str:
        raise NotImplementedError

    def get_supporting_facts(self, sample: dict) -> list[ChunkInfo]:
        raise NotImplementedError

    def get_chunks(self, sample: dict) -> list[ChunkInfo]:
        raise NotImplementedError

    def load_data(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def process_chunk(self, title: str, chunk: Union[str, list[str]]) -> str:
        if isinstance(chunk, list):
            chunk = " ".join(chunk)
        return f"{title}: {chunk}"
        # Title of chunk: chunk chunk chunk

    def collate_fn(self, batch: list[dict[str, Any]]):
        raise NotImplementedError


class HotpotQADataset(MultiHopDataset):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)

    def get_supporting_facts(self, sample: dict) -> list[ChunkInfo]:
        titles = list(set(sample["supporting_facts"]["title"]))
        supporting_facts = []
        for title in titles:
            idx = sample["context"]["title"].index(title)
            sentences = sample["context"]["sentences"][idx]
            chunk = self.process_chunk(title, sentences)
            info = ChunkInfo(idx, title, chunk)
            supporting_facts.append(info)
        return supporting_facts

    def get_chunks(self, sample: dict) -> list[ChunkInfo]:
        chunks = []
        for idx, (title, context) in enumerate(
            zip(sample["context"]["title"], sample["context"]["sentences"])
        ):
            chunk = self.process_chunk(title, context)
            info = ChunkInfo(idx, title, chunk)
            chunks.append(info)
        return chunks

    def get_type(self, sample: dict) -> str:
        type = sample["type"]
        if type == "bridge":
            return "compose"
        return type

    def get_hop(self, sample: dict) -> str:
        return 2


class WikiMQADataset(MultiHopDataset):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)

    def get_supporting_facts(self, sample: dict) -> list[ChunkInfo]:
        facts = sample["supporting_facts"]
        titles = list(set([fact[0] for fact in facts]))
        chunks = self.get_chunks(sample)
        facts = [chunk for chunk in chunks if chunk.title in titles]
        return facts

    def get_chunks(self, sample: dict) -> list[ChunkInfo]:
        chunks = []
        for idx, content in enumerate(sample["context"]):
            title = content[0]
            chunk = self.process_chunk(content[0], content[1])
            info = ChunkInfo(idx, title, chunk)
            chunks.append(info)
        return chunks

    def get_type(self, sample: dict) -> str:
        return sample["type"]

    def get_hop(self, sample: dict) -> str:
        if self.get_type(sample) == "bridge_compare":
            return 4
        return 2

    def get_id(self, sample: dict) -> str:
        return sample["_id"]

    def get_decomposition(self, sample: dict):
        chunks = self.get_chunks(sample)
        decompositions = []
        if len(sample["supporting_facts"]) != len(sample["evidences"]):
            return []
        for supporting_fact, evidence in zip(
            sample["supporting_facts"], sample["evidences"]
        ):
            title, idx = supporting_fact[0], supporting_fact[1]
            evidence = f"{evidence[0]} - {evidence[1]} - {evidence[2]}"
            fact_chunk = next(chunk for chunk in chunks if chunk.title == title)
            decomposition = {
                # 'chunk': chunks[idx].chunk,
                "id": fact_chunk.id,
                "chunk": fact_chunk.chunk,
                "title": title,
                "evidence": evidence,
            }
            decompositions.append(decomposition)
        return decompositions


class MuSiQueDataset(MultiHopDataset):
    def __init__(self, data_path) -> None:
        super().__init__(data_path)

    def get_supporting_facts(self, sample: dict) -> list[ChunkInfo]:
        chunks = self.get_chunks(sample)
        is_supports = sample["paragraphs"]["is_supporting"]
        supporting_facts = []
        for is_support, chunk in zip(is_supports, chunks):
            if is_support:
                supporting_facts.append(chunk)
        return supporting_facts

    def get_chunks(self, sample: dict) -> list[ChunkInfo]:
        chunks = []
        for idx, title, chunk in zip(
            sample["paragraphs"]["idx"],
            sample["paragraphs"]["title"],
            sample["paragraphs"]["paragraph_text"],
        ):
            chunk = self.process_chunk(title, chunk)
            info = ChunkInfo(idx, title, chunk)
            chunks.append(info)
        return chunks

    def get_decomposition(self, sample: dict) -> list[str]:
        return sample["question_decomposition"]

    def get_type(self, sample: dict) -> str:
        # TODO this is a difficult question
        return None

    def get_hop(self, sample: dict) -> str:
        return int(sample[id][0])


def get_dataset(dataset: AllDatasets, split: str) -> MultiHopDataset:
    if dataset == "hotpotQA":
        return HotpotQADataset(os.path.join(DATASET_PATH, "hotpotQA", f"{split}.json"))
    elif dataset == "2WikiMQA":
        return WikiMQADataset(os.path.join(DATASET_PATH, "2WikiMQA", f"{split}.json"))
    elif "musique" in dataset:
        return MuSiQueDataset(os.path.join(DATASET_PATH, dataset, f"{split}.json"))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    dataset = get_dataset("2WikiMQA", "demo")
    pprint(dataset[0])


if __name__ == "__main__":
    main()
