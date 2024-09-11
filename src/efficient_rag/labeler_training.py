import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import DataParallel
from transformers import DebertaV2Tokenizer, EvalPrediction, Trainer, TrainingArguments

from conf import (
    EFFICIENT_RAG_LABELER_TRAINING_DATA_PATH,
    MODEL_PATH,
    TAG_MAPPING,
    TAG_MAPPING_TWO,
    TERMINATE_ID,
)
from efficient_rag.data import LabelerDataset
from efficient_rag.model import DebertaForSequenceTokenClassification
from utils import load_jsonl


class LabelerTrainer(Trainer):
    def compute_loss(
        self,
        model: DebertaForSequenceTokenClassification,
        inputs: dict,
        return_outputs: bool = False,
    ):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        token_labels = inputs.pop("token_labels")
        sequence_labels = inputs.pop("sequence_labels")
        outputs = model(**inputs)
        token_logits = outputs.token_logits
        sequence_logits = outputs.sequence_logits

        weight = torch.tensor(WEIGHT_AVERAGE).cuda() # noqa
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        selected_sequence_logits = sequence_logits.argmax(-1)

        # Remove the token logits that labeled as <TERMINATE>
        token_logits = token_logits[selected_sequence_logits != TERMINATE_ID]
        token_labels = token_labels[selected_sequence_logits != TERMINATE_ID]

        module = model
        if type(model) == DataParallel:
            module = model.module

        token_loss = loss_fct(
            token_logits.view(-1, module.token_labels),
            token_labels.view(-1),
        )
        sequence_loss = loss_fct(
            sequence_logits.view(-1, module.sequence_labels),
            sequence_labels.view(-1),
        )
        wandb.log({"token_loss": token_loss, "sequence_loss": sequence_loss})
        loss = token_loss + sequence_loss
        return (loss, outputs) if return_outputs else loss


def eval_labeler(pred: EvalPrediction):
    tag_prediction = torch.tensor(pred.predictions[0].argmax(-1))
    token_prediction = torch.tensor(pred.predictions[1].argmax(-1))
    tag_label = torch.tensor(pred.label_ids[1])
    token_label = torch.tensor(pred.label_ids[0])

    mask = torch.tensor(pred.inputs != 0)
    token_prediction = torch.masked_select(token_prediction, mask)
    token_label = torch.masked_select(token_label, mask)

    result = {
        "tag_accuracy": accuracy_score(tag_prediction, tag_label),
        "token_accuracy": accuracy_score(token_prediction, token_label),
    }
    metrics = {
        # "recall": recall_score,
        # "precision": precision_score,
        "f1": f1_score,
    }

    tag_f1 = f1_score(tag_prediction, tag_label, average=None, zero_division=0)
    for tag, idx in CHUNK_TAG_MAPPING.items():
        result[f"tag_f1-{tag.strip('<>')}"] = tag_f1[idx]
    result["tag_f1"] = f1_score(token_prediction, token_label, average="micro")

    token_f1 = f1_score(tag_prediction, tag_label, average=None)
    result["token_f1_positive"] = token_f1[1]
    result["token_f1_negative"] = token_f1[0]
    result["token_f1"] = f1_score(tag_prediction, tag_label, average="micro")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientRAG Query Labeler")
    parser.add_argument(
        "--tags", type=int, default=2, choices=[2, 3], help="number of tags"
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--lr", help="learning rate", default=5e-6, type=float)
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--weight_average", action="store_true")
    args = parser.parse_args()
    return args


def build_dataset(
    dataset: str,
    split: str,
    max_len: int = 128,
    tokenizer=None,
    test_mode: bool = False,
    test_sample_cnt: int = 100,
):
    data_path = os.path.join(
        EFFICIENT_RAG_LABELER_TRAINING_DATA_PATH, dataset, f"{split}.jsonl"
    )
    data = load_jsonl(data_path)
    original_question = [d["question"] for d in data]
    chunk_tokens = [d["chunk_tokens"] for d in data]
    chunk_labels = [d["labels"] for d in data]
    tags = [CHUNK_TAG_MAPPING[d["tag"]] for d in data]

    if test_mode:
        return LabelerDataset(
            original_question[:test_sample_cnt],
            chunk_tokens[:test_sample_cnt],
            chunk_labels[:test_sample_cnt],
            tags[:test_sample_cnt],
            max_len,
            tokenizer,
        )
    return LabelerDataset(
        original_question, chunk_tokens, chunk_labels, tags, max_len, tokenizer
    )


def main(opt: argparse.Namespace):
    global tokenizer
    global CHUNK_TAG_MAPPING
    if opt.tags == 2:
        WANDB_PROJ_NAME = "EfficientRAG_labeler_two"
        CHUNK_TAG_MAPPING = TAG_MAPPING_TWO
    elif opt.tags == 3:
        WANDB_PROJ_NAME = "EfficientRAG_labeler"
        CHUNK_TAG_MAPPING = TAG_MAPPING
    os.environ["WANDB_PROJECT"] = WANDB_PROJ_NAME

    global WEIGHT_AVERAGE
    if opt.weight_average:
        # TODO: Add hotpotQA and 2WikiMQA weight here, noqa
        WEIGHT_AVERAGE = {
            "hotpotQA": [1.0, 1.0],
            "2WikiMQA": [0.51, 25.32],
            "musique": [0.51, 25.45],
        }[opt.dataset]
    else:
        WEIGHT_AVERAGE = [1.0, 1.0]


    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    model = DebertaForSequenceTokenClassification.from_pretrained(
        MODEL_PATH, sequence_labels=opt.tags, token_labels=2
    )
    save_path_mapping = {
        2: "saved_models/labeler_two",
        3: "saved_models/labeler",
    }
    save_dir = os.path.join(
        save_path_mapping[opt.tags],
        f"labeler_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}",
    )
    run_name = f"{opt.dataset}-{datetime.now().strftime(r'%m%d%H%M')}"
    train_dataset = build_dataset(
        opt.dataset,
        "train",
        opt.max_length,
        tokenizer,
        test_mode=opt.test,
        test_sample_cnt=opt.test_samples,
    )
    valid_dataset = build_dataset(
        opt.dataset,
        "valid",
        opt.max_length,
        tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=opt.epoch,
        learning_rate=opt.lr,
        per_device_train_batch_size=opt.batch_size,
        per_device_eval_batch_size=96,
        weight_decay=0.01,
        logging_dir=os.path.join(save_dir, "log"),
        save_strategy="epoch" if not opt.test else "no",
        evaluation_strategy="steps",
        eval_steps=opt.eval_steps,
        report_to="wandb",
        run_name=run_name,
        logging_steps=opt.logging_steps,
        warmup_steps=opt.warmup_steps,
        save_only_model=True,
        include_inputs_for_metrics=True,
    )
    trainer = LabelerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=eval_labeler,
    )
    trainer.train()


if __name__ == "__main__":
    options = parse_args()
    if options.test:
        os.environ["WANDB_MODE"] = "dryrun"
    main(options)
