import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    DebertaV2ForTokenClassification,
    DebertaV2Tokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from conf import EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH, MODEL_PATH
from efficient_rag.data import FilterDataset
from utils import load_jsonl

os.environ["WANDB_PROJECT"] = "EfficientRAG_filter"


def eval_filter(pred: EvalPrediction):
    preds = torch.tensor(pred.predictions.argmax(-1))
    labels = torch.tensor(pred.label_ids)
    mask = torch.tensor(pred.inputs != 0)

    preds = torch.masked_select(preds, mask)
    labels = torch.masked_select(labels, mask)

    filter_f1 = f1_score(labels, preds, average=None) #noqa

    result = {
        "accuracy": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds, average="micro"),
        "precision": precision_score(labels, preds, average="micro"),
        "f1": f1_score(labels, preds, average="micro"),
        "f1_marco": f1_score(labels, preds, average="macro"),
        "negative_f1": filter_f1[0],
        "positive_f1": filter_f1[1],
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientRAG Query Filter")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--save_path", type=str, default="saved_models/filter")
    parser.add_argument("--lr", help="learning rate", default=1e-5, type=float)
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    return args


def build_dataset(dataset: str, split: str, max_len: int = 128, tokenizer=None, test_mode=False):
    data_path = os.path.join(EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH, dataset, f"{split}.jsonl")
    data = load_jsonl(data_path)
    texts = [d["query_info_tokens"] for d in data]
    labels = [d["query_info_labels"] for d in data]
    if test_mode:
        return FilterDataset(texts[:100], labels[:100], max_len, tokenizer=tokenizer)
    return FilterDataset(texts, labels, max_len, tokenizer=tokenizer)


def main(opt: argparse.Namespace):
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    model = DebertaV2ForTokenClassification.from_pretrained(MODEL_PATH, num_labels=2)
    save_dir = os.path.join(opt.save_path, f"filter_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}")
    run_name = f"{opt.dataset}-{datetime.now().strftime(r'%m%d%H%M')}"
    train_dataset = build_dataset(opt.dataset, "train", opt.max_length, tokenizer, test_mode=opt.test)
    valid_dataset = build_dataset(opt.dataset, "valid", opt.max_length, tokenizer, test_mode=opt.test)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=opt.epoch,
        learning_rate=opt.lr,
        per_device_train_batch_size=opt.batch_size,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        logging_dir=os.path.join(save_dir, "log"),
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=opt.eval_steps,
        report_to="wandb",
        run_name=run_name,
        logging_steps=opt.logging_steps,
        warmup_steps=opt.warmup_steps,
        save_only_model=True,
        include_inputs_for_metrics=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=eval_filter,
    )
    trainer.train()


if __name__ == "__main__":
    options = parse_args()
    if options.test:
        os.environ["WANDB_MODE"] = "dryrun"
    main(options)
