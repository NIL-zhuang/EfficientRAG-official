import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm

from conf import CORPUS_DATA_PATH
from data_module import MultiHopDataset, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["musique", "2WikiMQA", "hotpotQA"],
        default="2WikiMQA"
    )
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()
    return args


def parse_chunks(dataset: MultiHopDataset):
    for sample in dataset:
        id = sample["id"]
        for idx, chunk in enumerate(sample["chunks"]):
            cid = f"{id}-{idx:02d}"
            yield {"id": cid, "text": chunk}


def purify_text(text: str):
    # delete all space and punctuations of the text
    pattern = r"[^\w]"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def merge_chunks(chunks: list[dict]):
    chunk_mapping = defaultdict(set)
    pattern_title = r"<title>(.*?)</title>"

    for chunk in chunks:
        cid = chunk["id"]
        text = chunk["text"]
        key = purify_text(text)
        # chunk_mapping[text].add(cid)
        chunk_mapping[key].add((cid, text))

    chunks = []
    # for text, ids in chunk_mapping.items():
    for key, id_text_pairs in tqdm(chunk_mapping.items()):
        id_text_pairs = list(id_text_pairs)
        text = id_text_pairs[0][1]
        ids = [pair[0] for pair in id_text_pairs]
        ids = "//".join(list(ids))
        title = text.split(":")[0].strip()
        chunk_info = {"id": ids, "title": title, "text": text}
        chunks.append(chunk_info)
    return chunks


def main(opt: argparse.Namespace):
    if opt.split is not None:
        split = [opt.split]
    else:
        split = ["train", "valid", "test"]
    chunks = []
    for s in split:
        try:
            dataset = get_dataset(opt.dataset, s)
            for d in parse_chunks(dataset):
                chunks.append(d)
        except:
            continue
    chunks = merge_chunks(chunks)
    output_dir = os.path.join(CORPUS_DATA_PATH, opt.dataset, "corpus.jsonl")
    with open(output_dir, "w+") as f:
        for chunk in chunks:
            data = json.dumps(chunk)
            f.write(data + "\n")


if __name__ == "__main__":
    options = parse_args()
    main(options)
