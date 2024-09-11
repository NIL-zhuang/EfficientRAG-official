import torch
from torch.utils.data import Dataset


class FilterDataset(Dataset):
    def __init__(
        self,
        texts: list[list[str]],
        labels: list[list[bool]] = None,
        max_len: int = 128,
        tokenizer=None,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.texts = texts
        self.labels = labels

        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"

    def __getitem__(self, index):
        text = self.texts[index]
        labels = self.labels[index][:]
        tokenized_text, labels = self.tokenize_and_preserve_labels(
            text, labels, self.tokenizer
        )
        assert len(tokenized_text) == len(labels)
        # [CLS] + tokenized_text + [SEP]
        labels = [False] + labels + [False]
        tokenized_text = [self.cls_token] + tokenized_text + [self.sep_token]

        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[: self.max_len]
            labels = labels[: self.max_len]
        else:
            append_length = self.max_len - len(tokenized_text)
            tokenized_text = tokenized_text + [self.pad_token] * append_length
            labels = labels + [False] * append_length

        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        sample = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return sample

    def __len__(self):
        return len(self.texts)

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
