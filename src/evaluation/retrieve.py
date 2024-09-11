import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import load_jsonl


def recall(oracle, chunks):
    match = [
        chunk for chunk in chunks if set(chunk.split("//")).intersection(set(oracle))
    ]
    match = list(set(match))
    recall = len(match) / len(oracle)
    return recall


def main(fpath: str):
    data = load_jsonl(fpath)

    result_avg_len = [len(sample["chunk_ids"]) for sample in data]
    print(f"Average number of chunks: {np.mean(result_avg_len)}")

    recall_list = [recall(sample["oracle_ids"], sample["chunk_ids"]) for sample in data]
    print(f"Recall: {np.mean(recall_list):.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    main(options.fpath)
