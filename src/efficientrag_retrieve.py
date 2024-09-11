import argparse
import json
import os
import sys
from typing import Iterator

import spacy
from tqdm.rich import tqdm_rich
from transformers import (
    DebertaV2ForTokenClassification,
    DebertaV2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from conf import (
    CLS_TOKEN,
    CONTINUE_TAG,
    CORPUS_DATA_PATH,
    FINISH_TAG,
    MODEL_PATH,
    RETRIEVE_RESULT_PATH,
    SEP_TOKEN,
    SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
    TAG_MAPPING_REV,
    TAG_MAPPING_TWO_REV,
    TERMINATE_TAG,
)
from data_module.format import build_query_info_sentence
from efficient_rag.model import DebertaForSequenceTokenClassification
from retrievers import Retriever
from utils import load_jsonl, write_jsonl

MAX_ITER = 4
LABELER_MAX_LENGTH = 384
FILTER_MAX_LENGTH = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--labels", type=int, default=2)
    parser.add_argument("--labeler_ckpt", type=str, required=True)
    parser.add_argument("--filter_ckpt", type=str, required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def spacify(
    text: str, nlp: spacy.Language = None, ignore_tokens=set([","])
) -> list[str]:
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    words = []
    for word in doc:
        if word.lemma_ not in ignore_tokens:
            words.append(word.text)
    return words


def tokenize_words(words: list[str], tokenizer: PreTrainedTokenizer):
    tokenized_text = sum([tokenizer.tokenize(word) for word in words], [])
    return tokenized_text


def build_labeler_input(
    query: str,
    chunks: list[str],
    tokenizer: PreTrainedTokenizer,
    nlp: spacy.Language,
):
    # construct inpus
    query_tokens = spacify(query, nlp)
    chunk_tokens = [spacify(chunk, nlp) for chunk in chunks]
    query_chunk_tokens = [
        [CLS_TOKEN] + query_tokens + [SEP_TOKEN] + chunk_token + [SEP_TOKEN]
        for chunk_token in chunk_tokens
    ]
    # from spacy to tokenizer
    query_chunk_tokens = [
        tokenize_words(query_chunk_token, tokenizer)
        for query_chunk_token in query_chunk_tokens
    ]
    # max length truncation
    query_chunk_tokens = [
        query_chunk_token[:LABELER_MAX_LENGTH]
        for query_chunk_token in query_chunk_tokens
    ]
    input_ids = [
        tokenizer.convert_tokens_to_ids(query_chunk_token)
        for query_chunk_token in query_chunk_tokens
    ]
    input_ids = {"input_ids": input_ids}
    tokenized = tokenizer.pad(
        input_ids,
        max_length=LABELER_MAX_LENGTH,
        return_attention_mask=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tokenized


def build_filter_input(
    query_info: str, tokenizer: PreTrainedTokenizer, nlp: spacy.Language = None
):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    query_info = [CLS_TOKEN] + spacify(query_info, nlp) + [SEP_TOKEN]
    query_info = tokenize_words(query_info, tokenizer)
    input_ids = tokenizer.convert_tokens_to_ids(query_info)
    input_ids = input_ids[:FILTER_MAX_LENGTH]
    input_ids = {"input_ids": [input_ids]}
    tokenized = tokenizer.pad(
        input_ids,
        max_length=FILTER_MAX_LENGTH,
        return_attention_mask=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tokenized


def label_info(
    labeler: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    nlp: spacy.Language,
    query: str,
    chunks: list[str],
):
    tokenized = build_labeler_input(query, chunks, tokenizer, nlp)
    tokenized_cuda = {k: v.cuda() for k, v in tokenized.items()}
    outputs_cuda = labeler(**tokenized_cuda)
    outputs = {k: v.cpu().detach().numpy() for k, v in outputs_cuda.items()}
    sequence_logits = outputs["sequence_logits"].argmax(-1)
    token_logits = outputs["token_logits"].argmax(-1)

    labeled_tokens = [
        tokenized["input_ids"][i][token_logits[i] == 1]
        for i in range(len(token_logits))
    ]
    infos = tokenizer.batch_decode(labeled_tokens, skip_special_tokens=True)
    if labeler.sequence_labels == 2:
        tag_mapping_rev = TAG_MAPPING_TWO_REV
    elif labeler.sequence_labels == 3:
        tag_mapping_rev = TAG_MAPPING_REV
    labels = [tag_mapping_rev[label] for label in sequence_logits]
    return infos, labels


def filter_query(
    filter: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    nlp: spacy.Language,
    prev_query: str,
    info_list: list[str],
) -> tuple[str, str]:
    info_list = [info for info in info_list if len(info.strip()) > 0]
    sentence = build_query_info_sentence(info_list, prev_query)
    tokenized = build_filter_input(sentence, tokenizer, nlp)
    tokenized_cuda = {k: v.cuda() for k, v in tokenized.items()}
    output_cuda = filter(**tokenized_cuda)
    output = {k: v.cpu().detach().numpy() for k, v in output_cuda.items()}
    token_logit = output["logits"].argmax(-1)
    filtered_ids = tokenized["input_ids"][token_logit == 1]
    filtered_query = tokenizer.decode(filtered_ids, skip_special_tokens=True)
    return sentence, filtered_query


def efficient_rag(
    labeler: PreTrainedModel,
    filter: PreTrainedModel,
    retriever: Retriever,
    tokenizer: PreTrainedTokenizer,
    dataset: list[dict],
    top_k: int = 10,
) -> Iterator[dict]:
    nlp = spacy.load("en_core_web_sm")
    for sample in tqdm_rich(dataset):
    # for sample in dataset:
        iter = 0
        filter_input = ""
        query = sample["question"]
        sample_chunks = {
            "query": query,
            "answer": sample["answer"],
            "oracle": [
                f"{sample['id']}-{'{:02d}'.format(chunk['positive_paragraph_idx'])}"
                for chunk in sample["decomposed_questions"].values()
            ],
            "oracle_docs": [
                chunk["positive_paragraph"]
                for chunk in sample["decomposed_questions"].values()
            ],
        }
        while iter < MAX_ITER:
            next_query_info = []
            chunks = retriever.search(query, top_k=top_k)[0]
            chunk_texts = [chunk["text"] for chunk in chunks]
            infos, labels = label_info(labeler, tokenizer, nlp, query, chunk_texts)
            sample_chunks[iter] = {
                "query": query,
                "filter_input": filter_input,
                "docs": [],
            }
            for chunk_info, label, chunk in zip(infos, labels, chunks):
                sample_chunk = {
                    "id": chunk["id"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "label": label,
                    "info": chunk_info,
                }
                sample_chunks[iter]["docs"].append(sample_chunk)

                if label == CONTINUE_TAG:
                    next_query_info.append(chunk_info)
                elif label in (TERMINATE_TAG, FINISH_TAG):
                    continue

            if len(next_query_info) == 0:
                break
            filter_input, query = filter_query(
                filter, tokenizer, nlp, query, next_query_info
            )
            iter += 1
        yield sample_chunks


def main(opt: argparse.Namespace):
    labeler = (
        DebertaForSequenceTokenClassification.from_pretrained(
            opt.labeler_ckpt, token_labels=2, sequence_labels=opt.labels
        )
        .cuda()
        .eval()
    )
    filter = (
        DebertaV2ForTokenClassification.from_pretrained(opt.filter_ckpt, num_labels=2)
        .cuda()
        .eval()
    )
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    passage_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "corpus.jsonl")
    embedding_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, opt.retriever)
    retriever = Retriever(
        passage_path=passage_path,
        passage_embedding_path=embedding_path,
        index_path_dir=embedding_path,
        model_type=opt.retriever,
    )
    dataset = load_jsonl(
        os.path.join(
            SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH, opt.dataset, "valid.jsonl"
            # SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH, opt.dataset, "demo.jsonl"
        )
    )
    if opt.test:
        dataset = dataset[:100]

    output_path = os.path.join(
        RETRIEVE_RESULT_PATH, "efficient_rag", f"{opt.dataset}-{opt.suffix}.jsonl"
    )
    import time
    start = time.time()
    with open(output_path, "w+", encoding="utf-8") as f:
        for chunk in efficient_rag(
            labeler, filter, retriever, tokenizer, dataset, top_k=opt.topk
        ):
            d = json.dumps(chunk, ensure_ascii=False)
            f.write(d + "\n")
    end = time.time()
    print(f"Retrieval time: {end - start:.2f}s")
    print(f"Average retrieval time: {(end - start) / len(dataset):.2f}s")


if __name__ == "__main__":
    options = parse_args()
    main(options)
