#!/usr/bin/env python3

"""
Dataset classes and data loader utilities for multitask training.

Call `load_multitask_data` to get (train/dev) examples for SST, Quora, STS, and ETPC.
"""

import csv
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from tokenizer import BertTokenizer


def preprocess_string(s: str) -> str:
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


# -----------------------
# Helpers for flexible CSV parsing
# -----------------------

_SENT1_KEYS = ("sentence1", "sent1", "s1", "text1", "question1", "premise", "left", "source", "a")
_SENT2_KEYS = ("sentence2", "sent2", "s2", "text2", "question2", "hypothesis", "right", "target", "b")
_ID_KEYS = ("id", "pair_id", "pairid", "guid", "qid", "index")


def _first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return d[k]
    return None


def _to_int01(v: Any) -> int:
    """Convert common truthy/falsey string/number values to 0/1."""
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    if s in {"0", "0.0", "false", "f", "no", "n", ""}:
        return 0
    # try numeric
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def _one_hot(index: int, num_classes: int) -> List[int]:
    vec = [0] * num_classes
    if 0 <= index < num_classes:
        vec[index] = 1
    return vec


def _parse_etpc_labels(record: Dict[str, Any], known_classes: int = 7) -> Tuple[List[int], bool]:
    """
    Try multiple label formats:
      - 7 separate binary columns (any names except id/sentence columns)
      - single numeric 'label' (0..6) -> one-hot
      - single string label -> ignore (return zeros)
    Returns (labels, warned) where warned=True if labels were synthesized as zeros.
    """
    # Normalize keys to lower for filtering, but keep original dict for values
    keys_lower = {k.lower(): k for k in record.keys()}

    # Base (non-label) fields we should ignore as labels
    base_candidates = set(_ID_KEYS) | set(_SENT1_KEYS) | set(_SENT2_KEYS) | {"split", "domain"}

    # Case A: explicit multi-label columns = all fields not in base set
    label_field_candidates = [orig for lower, orig in keys_lower.items() if lower not in base_candidates]

    # If there is exactly one label-like column named 'label' or similar, handle separately
    single_label_key = None
    for name in ("label", "gold_label", "class", "y"):
        if name in keys_lower:
            single_label_key = keys_lower[name]
            break

    warned = False
    labels: List[int] = []

    if single_label_key is not None and (len(label_field_candidates) == 0 or single_label_key in label_field_candidates):
        # Single label value -> attempt numeric -> one-hot
        raw = record[single_label_key]
        try:
            idx = int(float(raw))
            labels = _one_hot(idx, known_classes)
            return labels, warned
        except Exception:
            # Could be a string label we don't recognize: fallback to zeros
            labels = [0] * known_classes
            warned = True
            return labels, warned

    if len(label_field_candidates) > 0:
        # Build binary vector from all candidate columns (truncate/pad to 7)
        labels = [_to_int01(record[k]) for k in label_field_candidates]
        if len(labels) < known_classes:
            labels = labels + [0] * (known_classes - len(labels))
        elif len(labels) > known_classes:
            labels = labels[:known_classes]
        return labels, warned

    # No label columns found at all -> zeros
    labels = [0] * known_classes
    warned = True
    return labels, warned


# -----------------------
# Datasets
# -----------------------

class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)
        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sents": sents,
            "sent_ids": sent_ids,
        }


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)
        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "sents": sents,
            "sent_ids": sent_ids,
        }


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression: bool = False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]

        # Accept both (s1, s2, label/s, sent_id) and (s1, s2, sent_id)
        has_labels = len(data[0]) >= 4 and not isinstance(data[0][2], str)
        if has_labels:
            labels = [x[2] for x in data]  # could be scalar or list (multi-label)
            sent_ids = [x[3] for x in data]
        else:
            labels = None
            sent_ids = [x[-1] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        ttids1 = encoding1["token_type_ids"] if "token_type_ids" in encoding1 else torch.zeros_like(token_ids)

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        ttids2 = encoding2["token_type_ids"] if "token_type_ids" in encoding2 else torch.zeros_like(token_ids2)

        if labels is None:
            # Placeholder labels to avoid crashes in inference-only scenarios
            labels_tensor = torch.zeros(
                len(sent_ids),
                dtype=torch.double if self.isRegression else torch.long,
            )
        else:
            # Choose dtype based on task: regression -> double, multi-label -> float, else -> long
            if self.isRegression:
                labels_tensor = torch.as_tensor(labels, dtype=torch.double)
            else:
                # Multi-label if each element is a sequence
                is_multilabel = isinstance(labels[0], (list, tuple))
                labels_tensor = torch.as_tensor(labels, dtype=torch.float32 if is_multilabel else torch.long)

        return (
            token_ids,
            ttids1.long(),
            attention_mask,
            token_ids2,
            ttids2.long(),
            attention_mask2,
            labels_tensor,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        ) = self.pad_data(all_data)

        return {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
            "sent_ids": sent_ids,
        }


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        ttids1 = encoding1["token_type_ids"] if "token_type_ids" in encoding1 else torch.zeros_like(token_ids)

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        ttids2 = encoding2["token_type_ids"] if "token_type_ids" in encoding2 else torch.zeros_like(token_ids2)

        return (
            token_ids,
            ttids1.long(),
            attention_mask,
            token_ids2,
            ttids2.long(),
            attention_mask2,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            sent_ids,
        ) = self.pad_data(all_data)

        return {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "sent_ids": sent_ids,
        }


# -----------------------
# Loader
# -----------------------

def load_multitask_data(sst_filename, quora_filename, sts_filename, etpc_filename, split="train"):
    # ---- SST ----
    sst_data: List[Tuple] = []
    num_labels: Dict[int, int] = {}
    if split == "test":
        with open(sst_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter=","):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                sst_data.append((sent, sent_id))
    else:
        with open(sst_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter=","):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                label = int(float(record["sentiment"].strip()))
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sst_data.append((sent, label, sent_id))
    print(f"Loaded {len(sst_data)} {split} examples from {sst_filename}")

    # ---- Quora ----
    quora_data: List[Tuple] = []
    if split == "test":
        with open(quora_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter=","):
                sent_id = record["id"].lower().strip()
                quora_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )
    else:
        with open(quora_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter=","):
                try:
                    sent_id = record["id"].lower().strip()
                    quora_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            int(float(record["is_duplicate"])),
                            sent_id,
                        )
                    )
                except Exception:
                    # Skip malformed rows
                    continue
    print(f"Loaded {len(quora_data)} {split} examples from {quora_filename}")

    # ---- STS ----
    sts_data: List[Tuple] = []
    if split == "test":
        with open(sts_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter=","):
                sent_id = record["id"].lower().strip()
                sts_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )
    else:
        with open(sts_filename, "r", encoding="utf-8") as fp:
            for record in csv.DictReader(fp, delimiter=","):
                sent_id = record["id"].lower().strip()
                sts_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        float(record["similarity"]),
                        sent_id,
                    )
                )
    print(f"Loaded {len(sts_data)} {split} examples from {sts_filename}")

    # ---- ETPC ----
    etpc_data: List[Tuple] = []
    etpc_warned_missing_labels = False

    with open(etpc_filename, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter=",")
        for raw in reader:
            # Normalize a lower-cased view but keep original keys for values
            rec_lower = {k.lower(): v for k, v in raw.items()}

            s1 = _first_present(rec_lower, _SENT1_KEYS)
            s2 = _first_present(rec_lower, _SENT2_KEYS)
            sid = _first_present(rec_lower, _ID_KEYS) or ""

            if s1 is None or s2 is None:
                # Can't use this row; continue
                continue

            s1 = preprocess_string(s1)
            s2 = preprocess_string(s2)
            sid = str(sid).lower().strip()

            if split == "test":
                etpc_data.append((s1, s2, sid))
            else:
                labels, warned = _parse_etpc_labels(raw)  # pass original dict for exact keys
                etpc_warned_missing_labels = etpc_warned_missing_labels or warned
                etpc_data.append((s1, s2, labels, sid))

    print(f"Loaded {len(etpc_data)} {split} examples from {etpc_filename}")
    if split != "test" and etpc_warned_missing_labels:
        print(
            "[Warning] ETPC: No explicit label columns detected in some/all rows. "
            "Filled zeros as placeholders. Ensure your ETPC train/dev CSV contains labels."
        )

    return sst_data, num_labels, quora_data, sts_data, etpc_data
