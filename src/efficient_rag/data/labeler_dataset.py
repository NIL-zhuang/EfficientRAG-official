import spacy
import torch
from torch.utils.data import Dataset


class LabelerDataset(Dataset):
    def __init__(
        self,
        questions: list[str],
        chunk_tokens: list[list[str]],
        labels: list[list[str]],
        tags: list[int],
        max_len=512,
        tokenizer=None,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.questions = questions
        self.labels = labels
        self.chunk_tokens = chunk_tokens
        self.tags = tags

        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"

        self.nlp = spacy.load("en_core_web_sm")

    def __getitem__(self, index):
        # text = self.questions[index]
        chunk = self.chunk_tokens[index]
        chunk_label = self.labels[index][:]
        question, question_labels = self.construct_question_labels(self.questions[index])
        tokenized_question, tokenized_question_labels = self.tokenize_and_preserve_labels(
            question, question_labels, self.tokenizer
        )
        assert self.labels is not None

        tokenized_chunk, chunk_label = self.tokenize_and_preserve_labels(chunk, chunk_label, self.tokenizer)
        assert len(tokenized_chunk) == len(chunk_label)

        # [CLS] question [SEP] chunk [SEP]
        tokenized_text = [self.cls_token] + tokenized_question + [self.sep_token] + tokenized_chunk + [self.sep_token]
        labels = [False] + tokenized_question_labels + [False] + chunk_label + [False]

        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[: self.max_len]
            if self.labels is not None:
                labels = labels[: self.max_len]
        else:
            tokenized_text = tokenized_text + [self.pad_token for _ in range(self.max_len - len(tokenized_text))]
            if self.labels is not None:
                labels = labels + [False for _ in range(self.max_len - len(labels))]

        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        chunk_tags = self.tags[index]
        sample = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "token_labels": torch.tensor(labels, dtype=torch.long),
            "sequence_labels": torch.tensor(chunk_tags, dtype=torch.long),
        }
        return sample

    def __len__(self):
        return len(self.questions)

    def tokenize_and_preserve_labels(self, text, text_labels, tokenizer):
        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword. It is, of course, a bit slower in processing
        time, but it will help our model achieve higher accuracy.
        """

        tokenized_text = []
        labels = []
        for word, label in zip(text, text_labels):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_text.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        return tokenized_text, labels

    def construct_question_labels(self, question, ignore_tokens=set([","])):
        doc = self.nlp(question)
        words = []
        for word in doc:
            if word.lemma_ not in ignore_tokens:
                words.append(word.text)
        labels = [False] * len(words)
        return words, labels
