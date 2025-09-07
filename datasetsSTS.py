# datasetsSTS.py

import csv
import os
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _read_csv_flexible(path: str) -> List[Dict[str, Any]]:
    """
    Read a CSV file into a list of row dictionaries.

    This function is tolerant to missing files or empty CSVs and will
    return an empty list in those cases.

    Args:
        path (str): Path to the CSV file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, one per row.
    """
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        for r in reader:
            rows.append({(k or "").strip(): (v if v is not None else "") for k, v in r.items()})
    return rows


def _pick_first_present(d: Dict[str, Any], candidates: List[str], default=None):
    """
    Return the first non-empty value in a dictionary among candidate keys.

    Args:
        d (Dict[str, Any]): Source dictionary.
        candidates (List[str]): Keys to check in order.
        default (Any, optional): Value to return if none are present.

    Returns:
        Any: The first present value or the default.
    """
    for k in candidates:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _to_float_or_none(x) -> Optional[float]:
    """
    Convert a value to float if possible, otherwise return None.

    Args:
        x (Any): Input value.

    Returns:
        Optional[float]: Parsed float or None on failure.
    """
    try:
        return float(x)
    except Exception:
        return None


def _get_tokenizer(args):
    """
    Build a Hugging Face tokenizer from CLI args.

    Args:
        args (argparse.Namespace): Arguments with hf_model_name and local_files_only.

    Returns:
        transformers.PreTrainedTokenizerBase: Instantiated tokenizer.
    """
    name = getattr(args, "hf_model_name", "") or "bert-base-uncased"
    return AutoTokenizer.from_pretrained(name, local_files_only=getattr(args, "local_files_only", False))


def _max_len(args) -> int:
    """
    Get the maximum tokenized sequence length from args.

    Args:
        args (argparse.Namespace): Arguments possibly containing max_length.

    Returns:
        int: Maximum sequence length (defaults to 128).
    """
    return int(getattr(args, "max_length", 128))


def _pad_2d_long(seqs: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """
    Left-align and right-pad a list of 1D LongTensors to a 2D tensor.

    Args:
        seqs (List[torch.Tensor]): Sequence tensors of shape [L_i].
        pad_value (int): Padding value.

    Returns:
        torch.Tensor: Padded tensor of shape [B, max_L].
    """
    if len(seqs) == 0:
        return torch.empty(0, 0, dtype=torch.long)
    max_len = max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        if L > 0:
            out[i, :L] = s[:L]
    return out


class SentenceClassificationDataset(Dataset):
    """
    Single-sentence dataset for classification/regression tasks.

    Expected row fields (flexible): text in one of
    {text, sentence, sentence1, s1, review, phrase} and optional label.

    Args:
        data (List[Dict[str, Any]]): Normalized row dictionaries.
        args (argparse.Namespace): Tokenization and length settings.
    """

    def __init__(self, data: List[Dict[str, Any]], args):
        self.data = data or []
        self.args = args
        self.tokenizer = _get_tokenizer(args)
        self.max_len = _max_len(args)

    def __len__(self):
        """
        Return the number of examples.

        Returns:
            int: Dataset size.
        """
        return len(self.data)

    def _extract(self, row: Dict[str, Any]) -> Tuple[str, Optional[float], Optional[Any]]:
        """
        Extract text, label, and optional id from a raw row.

        Args:
            row (Dict[str, Any]): Source row dictionary.

        Returns:
            Tuple[str, Optional[float], Optional[Any]]: (text, label, sent_id)
        """
        text = _pick_first_present(
            row,
            ["text", "sentence", "sentence1", "s1", "review", "phrase"],
            default=""
        )
        label = _pick_first_present(row, ["label", "labels", "score", "y"])
        label = _to_float_or_none(label)
        sid = _pick_first_present(row, ["id", "sent_id", "pair_id", "guid"])
        return text, label, sid

    def __getitem__(self, idx):
        """
        Tokenize an example and return tensors suitable for collation.

        Args:
            idx (int): Index of the example.

        Returns:
            Dict[str, torch.Tensor | float | Any]: Token ids, attention mask, and optional label/id.
        """
        row = self.data[idx]
        text, label, sid = self._extract(row)
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors=None,
        )
        item = {
            "token_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }
        if label is not None:
            item["labels"] = float(label)
        if sid is not None:
            item["id"] = sid
        return item

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of items into padded tensors.

        Handles both dict items (preferred) and tuple items of the form
        (ids, mask[, label]).

        Args:
            batch (List[Any]): Sequence of dataset items.

        Returns:
            Dict[str, torch.Tensor]: Padded token ids/masks and label tensors.
        """
        ids, mask, labels, labels_mask = [], [], [], []

        for i, item in enumerate(batch):
            if isinstance(item, dict):
                tid = item["token_ids"]
                am = item["attention_mask"]
                lab = item.get("labels", None)
            else:
                tid = item[0]
                am = item[1]
                lab = item[2] if len(item) >= 3 else None

            tid = tid if torch.is_tensor(tid) else torch.as_tensor(tid, dtype=torch.long)
            am = am if torch.is_tensor(am) else torch.as_tensor(am, dtype=torch.long)
            ids.append(tid)
            mask.append(am)

            if lab is None:
                labels.append(0.0)
                labels_mask.append(0)
            else:
                labels.append(float(lab))
                labels_mask.append(1)

        return {
            "token_ids": _pad_2d_long(ids, pad_value=0),
            "attention_mask": _pad_2d_long(mask, pad_value=0),
            "labels": torch.tensor(labels, dtype=torch.float),
            "labels_mask": torch.tensor(labels_mask, dtype=torch.uint8),
        }


class SentencePairDataset(Dataset):
    """
    Sentence-pair dataset for QQP/STS/ETPC tasks.

    Expected row fields (flexible):
    - text1 in {text1, sentence1, s1, question1, q1, premise}
    - text2 in {text2, sentence2, s2, question2, q2, hypothesis}
    - optional label and id.

    Args:
        data (List[Dict[str, Any]]): Normalized row dictionaries.
        args (argparse.Namespace): Tokenization and length settings.
    """

    def __init__(self, data: List[Dict[str, Any]], args):
        self.data = data or []
        self.args = args
        self.tokenizer = _get_tokenizer(args)
        self.max_len = _max_len(args)

    def __len__(self):
        """
        Return the number of examples.

        Returns:
            int: Dataset size.
        """
        return len(self.data)

    def _extract_pair(self, row: Dict[str, Any]) -> Tuple[str, str, Optional[float], Optional[Any]]:
        """
        Extract sentence pair, label, and optional id from a raw row.

        Args:
            row (Dict[str, Any]): Source row dictionary.

        Returns:
            Tuple[str, str, Optional[float], Optional[Any]]: (text1, text2, label, sent_id)
        """
        t1 = _pick_first_present(row, ["text1", "sentence1", "s1", "question1", "q1", "premise"], default="")
        t2 = _pick_first_present(row, ["text2", "sentence2", "s2", "question2", "q2", "hypothesis"], default="")
        if not t2 and "text" in row and "text1" in row:
            t2 = row.get("text", "")
        label = _pick_first_present(row, ["label", "labels", "score", "similarity", "y"])
        label = _to_float_or_none(label)
        sid = _pick_first_present(row, ["id", "pair_id", "sent_id", "guid"])
        return t1, t2, label, sid

    def __getitem__(self, idx):
        """
        Tokenize a pair example and return tensors suitable for collation.

        Args:
            idx (int): Index of the example.

        Returns:
            Dict[str, torch.Tensor | float | Any]: Token ids/masks for both sentences and optional label/id.
        """
        row = self.data[idx]
        t1, t2, label, sid = self._extract_pair(row)

        enc1 = self.tokenizer(
            t1, truncation=True, padding=False, max_length=self.max_len,
            return_attention_mask=True, return_tensors=None,
        )
        enc2 = self.tokenizer(
            t2, truncation=True, padding=False, max_length=self.max_len,
            return_attention_mask=True, return_tensors=None,
        )
        item = {
            "token_ids_1": torch.tensor(enc1["input_ids"], dtype=torch.long),
            "attention_mask_1": torch.tensor(enc1["attention_mask"], dtype=torch.long),
            "token_ids_2": torch.tensor(enc2["input_ids"], dtype=torch.long),
            "attention_mask_2": torch.tensor(enc2["attention_mask"], dtype=torch.long),
        }
        if label is not None:
            item["labels"] = float(label)
        if sid is not None:
            item["id"] = sid
        return item

    def pad_data(self, data: List[Any]):
        """
        Pad a batch of sentence-pair items into uniform tensors.

        Accepts either dict items (preferred) or tuples of the form:
        (ids1, mask1, ids2, mask2[, label][, sent_id])

        Args:
            data (List[Any]): Sequence of dataset items.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Any], torch.Tensor]:
                token_ids_1, attention_mask_1, token_ids_2, attention_mask_2,
                labels, sent_ids(list of ids, not a tensor), labels_mask.
        """
        tok1, msk1, tok2, msk2, lbls, sids, lblmask = [], [], [], [], [], [], []

        for i, item in enumerate(data):
            if isinstance(item, dict):
                ids1 = item["token_ids_1"]
                am1 = item["attention_mask_1"]
                ids2 = item["token_ids_2"]
                am2 = item["attention_mask_2"]
                lab = item.get("labels", None)
                sid = item.get("id", None)
            else:
                if len(item) < 4:
                    raise ValueError(f"SentencePairDataset item too short: len={len(item)}")
                ids1, am1, ids2, am2 = item[0], item[1], item[2], item[3]
                lab = item[4] if len(item) >= 5 else None
                sid = item[5] if len(item) >= 6 else i

            ids1 = ids1 if torch.is_tensor(ids1) else torch.as_tensor(ids1, dtype=torch.long)
            am1 = am1 if torch.is_tensor(am1) else torch.as_tensor(am1, dtype=torch.long)
            ids2 = ids2 if torch.is_tensor(ids2) else torch.as_tensor(ids2, dtype=torch.long)
            am2 = am2 if torch.is_tensor(am2) else torch.as_tensor(am2, dtype=torch.long)

            tok1.append(ids1)
            msk1.append(am1)
            tok2.append(ids2)
            msk2.append(am2)

            if lab is None:
                lbls.append(0.0)
                lblmask.append(0)
            else:
                lbls.append(float(lab))
                lblmask.append(1)

            sids.append(str(sid) if sid is not None else str(i))
        token_ids_1 = _pad_2d_long(tok1, pad_value=0)
        attention_mask_1 = _pad_2d_long(msk1, pad_value=0)
        token_ids_2 = _pad_2d_long(tok2, pad_value=0)
        attention_mask_2 = _pad_2d_long(msk2, pad_value=0)
        labels = torch.tensor(lbls, dtype=torch.float)
        sent_ids = sids
        labels_mask = torch.tensor(lblmask, dtype=torch.uint8)

        return token_ids_1, attention_mask_1, token_ids_2, attention_mask_2, labels, sent_ids, labels_mask

    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        """
        Collate a batch of pair items using robust padding.

        Args:
            batch (List[Any]): Sequence of dataset items.

        Returns:
            Dict[str, Any]: Padded tensors and masks for the pair task, with
                'sent_ids' kept as a Python list (not a tensor).
        """
        (
            token_ids_1,
            attention_mask_1,
            token_ids_2,
            attention_mask_2,
            labels,
            sent_ids,
            labels_mask,
        ) = self.pad_data(batch)

        return {
            "token_ids_1": token_ids_1,
            "attention_mask_1": attention_mask_1,
            "token_ids_2": token_ids_2,
            "attention_mask_2": attention_mask_2,
            "labels": labels,
            "labels_mask": labels_mask,
            "sent_ids": sent_ids,
        }


def _normalize_sst_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize heterogeneous SST-like rows to a {text, label} schema.

    Args:
        rows (List[Dict[str, Any]]): Raw row dictionaries.

    Returns:
        List[Dict[str, Any]]: Normalized rows with keys {"text","label"}.
    """
    norm = []
    for r in rows:
        text = _pick_first_present(
            r, ["text", "sentence", "review", "phrase"],
            default=_pick_first_present(r, list(r.keys()), default="")
        )
        label = _pick_first_present(r, ["label", "labels", "score", "y"])
        sid = _pick_first_present(r, ["id", "pair_id", "sent_id", "guid"])
        norm.append({"text": text, "label": label, "id": sid})
    return norm


def _normalize_pair_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize heterogeneous pair rows to a {text1, text2, label} schema.

    Args:
        rows (List[Dict[str, Any]]): Raw row dictionaries.

    Returns:
        List[Dict[str, Any]]: Normalized rows with keys {"text1","text2","label"}.
    """
    norm = []
    for r in rows:
        t1 = _pick_first_present(
            r, ["text1", "sentence1", "s1", "question1", "q1", "premise"],
            default=_pick_first_present(r, list(r.keys()), default="")
        )
        t2 = _pick_first_present(r, ["text2", "sentence2", "s2", "question2", "q2", "hypothesis"], default="")
        if t2 == "" and "text" in r and "text1" in r:
            t2 = r.get("text", "")
        label = _pick_first_present(r, ["label", "labels", "score", "similarity", "y"])
        sid = _pick_first_present(r, ["id", "pair_id", "sent_id", "guid"])
        norm.append({"text1": t1, "text2": t2, "label": label, "id": sid})
    return norm


def load_multitask_data(sst_path: str, quora_path: str, sts_path: str, etpc_path: str, split: str):
    """
    Load and normalize CSVs for multiple tasks.

    The return signature preserves backward compatibility with existing code:
    it returns five values, where the second is a placeholder (None).

    Args:
        sst_path (str): Path to SST CSV.
        quora_path (str): Path to QQP CSV.
        sts_path (str): Path to STS CSV.
        etpc_path (str): Path to ETPC CSV.
        split (str): Unused split indicator kept for compatibility.

    Returns:
        Tuple[List[Dict[str, Any]], None, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
            (sst_data, None, quora_data, sts_data, etpc_data)
    """
    sst_rows = _read_csv_flexible(sst_path)
    quora_rows = _read_csv_flexible(quora_path)
    sts_rows = _read_csv_flexible(sts_path)
    etpc_rows = _read_csv_flexible(etpc_path)

    sst_data = _normalize_sst_rows(sst_rows)
    quora_data = _normalize_pair_rows(quora_rows)
    sts_data = _normalize_pair_rows(sts_rows)
    etpc_data = _normalize_pair_rows(etpc_rows)

    _placeholder = None
    return sst_data, _placeholder, quora_data, sts_data, etpc_data
