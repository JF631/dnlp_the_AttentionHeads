import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasetsSTS import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)

TQDM_DISABLE = False

def _binary_f1(y_true, y_pred):
    """
    Compute binary F1 score from 0/1 labels and predictions.

    Args:
        y_true (Iterable[int]): Ground-truth binary labels.
        y_pred (Iterable[int]): Predicted binary labels.

    Returns:
        float: F1 score in [0, 1].
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(2 * precision * recall / (precision + recall + 1e-12))


def _ensure_ids(sent_ids_tensor, batch_size, start_idx):
    """
    Convert an optional tensor/list of sentence IDs to a Python list.

    If the input is None, a consecutive range starting at `start_idx`
    with length `batch_size` is returned.

    Args:
        sent_ids_tensor (Optional[torch.Tensor | Iterable[int]]): Input ids or None.
        batch_size (int): Fallback length if ids are missing.
        start_idx (int): Starting index for generated ids.

    Returns:
        List[int]: Sentence ids for the batch.
    """
    if sent_ids_tensor is None:
        return list(range(start_idx, start_idx + batch_size))
    if isinstance(sent_ids_tensor, torch.Tensor):
        return sent_ids_tensor.detach().cpu().tolist()
    return list(sent_ids_tensor)


def model_eval_multitask(
    sst_dataloader, quora_dataloader, sts_dataloader, etpc_dataloader, model, device, task
):
    """
    Evaluate a multitask model on development splits (labels available).

    Computes task-appropriate metrics and returns per-task predictions and ids.

    Args:
        sst_dataloader (Optional[DataLoader]): SST dev loader.
        quora_dataloader (Optional[DataLoader]): QQP dev loader.
        sts_dataloader (Optional[DataLoader]): STS dev loader.
        etpc_dataloader (Optional[DataLoader]): ETPC dev loader.
        model (torch.nn.Module): Multitask model with task heads.
        device (torch.device): Inference device.
        task (str): One of {"sst","sts","qqp","etpc","multitask"}.

    Returns:
        Tuple:
            (
                quora_accuracy (Optional[float]),
                quora_f1 (Optional[float]),
                quora_y_pred (List[int]),
                quora_sent_ids (List[int]),
                sst_accuracy (Optional[float]),
                sst_y_pred (List[int]),
                sst_sent_ids (List[int]),
                sts_corr (Optional[float]),
                sts_y_pred (List[float]),
                sts_sent_ids (List[int]),
                etpc_accuracy (Optional[float]),
                etpc_y_pred (List[List[int]]),
                etpc_sent_ids (List[int]),
            )
    """
    model.eval()

    with torch.no_grad():
        quora_y_true, quora_y_pred, quora_sent_ids = [], [], []
        running_idx = 0
        if task in ("qqp", "multitask") and quora_dataloader is not None:
            for batch in tqdm(quora_dataloader, desc="eval", disable=TQDM_DISABLE):
                ids1 = batch["token_ids_1"].to(device)
                m1 = batch["attention_mask_1"].to(device)
                ids2 = batch["token_ids_2"].to(device)
                m2 = batch["attention_mask_2"].to(device)
                labels = batch.get("labels", None)
                sent_ids = batch.get("sent_ids", None)

                logits = model.predict_paraphrase(ids1, m1, ids2, m2)
                y_hat = logits.sigmoid().round().flatten().cpu().numpy()

                if labels is not None:
                    y_gold = labels.flatten().cpu().numpy()
                else:
                    y_gold = np.zeros_like(y_hat)

                quora_y_pred.extend(y_hat.tolist())
                quora_y_true.extend(y_gold.tolist())

                b_ids = _ensure_ids(sent_ids, ids1.size(0), running_idx)
                quora_sent_ids.extend(b_ids)
                running_idx += ids1.size(0)

            quora_accuracy = float(np.mean(np.array(quora_y_pred) == np.array(quora_y_true))) if quora_y_true else None
            quora_f1 = _binary_f1(quora_y_true, quora_y_pred) if quora_y_true else None
        else:
            quora_accuracy = None
            quora_f1 = None

        sts_y_true, sts_y_pred, sts_sent_ids = [], [], []
        running_idx = 0
        if task in ("sts", "multitask") and sts_dataloader is not None:
            for batch in tqdm(sts_dataloader, desc="eval", disable=TQDM_DISABLE):
                ids1 = batch["token_ids_1"].to(device)
                m1 = batch["attention_mask_1"].to(device)
                ids2 = batch["token_ids_2"].to(device)
                m2 = batch["attention_mask_2"].to(device)
                labels = batch.get("labels", None)
                sent_ids = batch.get("sent_ids", None)

                scores = model.predict_similarity(ids1, m1, ids2, m2).detach().cpu().numpy()
                y_hat = scores.flatten()

                if labels is not None:
                    y_gold = labels.flatten().cpu().numpy()
                else:
                    y_gold = np.zeros_like(y_hat)

                sts_y_pred.extend(y_hat.tolist())
                sts_y_true.extend(y_gold.tolist())

                b_ids = _ensure_ids(sent_ids, ids1.size(0), running_idx)
                print(b_ids)
                sts_sent_ids.extend(b_ids)
                running_idx += ids1.size(0)

            if len(sts_y_pred) >= 2 and len(sts_y_true) >= 2:
                pearson_mat = np.corrcoef(sts_y_pred, sts_y_true)
                sts_corr = float(pearson_mat[1][0])
            else:
                sts_corr = 0.0
        else:
            sts_corr = None

        sst_y_true, sst_y_pred, sst_sent_ids = [], [], []
        running_idx = 0
        if task in ("sst", "multitask") and sst_dataloader is not None:
            for batch in tqdm(sst_dataloader, desc="eval", disable=TQDM_DISABLE):
                ids = batch["token_ids"].to(device)
                m = batch["attention_mask"].to(device)
                labels = batch.get("labels", None)
                sent_ids = batch.get("sent_ids", None)

                logits = model.predict_sentiment(ids, m)
                y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

                if labels is not None:
                    y_gold = labels.flatten().cpu().numpy()
                else:
                    y_gold = np.zeros_like(y_hat)

                sst_y_pred.extend(y_hat.tolist())
                sst_y_true.extend(y_gold.tolist())

                b_ids = _ensure_ids(sent_ids, ids.size(0), running_idx)
                sst_sent_ids.extend(b_ids)
                running_idx += ids.size(0)

            sst_accuracy = float(np.mean(np.array(sst_y_pred) == np.array(sst_y_true))) if sst_y_true else None
        else:
            sst_accuracy = None

        etpc_y_true, etpc_y_pred, etpc_sent_ids = [], [], []
        running_idx = 0
        if task == "etpc" and etpc_dataloader is not None:
            for batch in tqdm(etpc_dataloader, desc="eval", disable=TQDM_DISABLE):
                ids1 = batch["token_ids_1"].to(device)
                m1 = batch["attention_mask_1"].to(device)
                ids2 = batch["token_ids_2"].to(device)
                m2 = batch["attention_mask_2"].to(device)
                labels = batch.get("labels", None)
                sent_ids = batch.get("sent_ids", None)

                logits = model.predict_paraphrase_types(ids1, m1, ids2, m2)
                y_hat = logits.sigmoid().round().to(torch.int).cpu().tolist()

                etpc_y_pred.extend(y_hat)
                if labels is not None:
                    y_true = labels.round().to(torch.int).cpu().tolist()
                    etpc_y_true.extend(y_true)

                b_ids = _ensure_ids(sent_ids, ids1.size(0), running_idx)
                etpc_sent_ids.extend(b_ids)
                running_idx += ids1.size(0)

            if etpc_y_true:
                pred_arr = np.array(etpc_y_pred, dtype=int)
                true_arr = np.array(etpc_y_true, dtype=int)
                correct_pred = np.all(pred_arr == true_arr, axis=1).astype(int)
                etpc_accuracy = float(np.mean(correct_pred))
            else:
                etpc_accuracy = None
        else:
            etpc_accuracy = None

        if task in ("qqp", "multitask") and quora_accuracy is not None:
            print(f"Paraphrase detection accuracy: {quora_accuracy:.3f}")
            print(f"Paraphrase detection F1: {quora_f1:.3f}")
        if task in ("sst", "multitask") and sst_accuracy is not None:
            print(f"Sentiment classification accuracy: {sst_accuracy:.3f}")
        if task in ("sts", "multitask") and sts_corr is not None:
            print(f"Semantic Textual Similarity Pearson r: {sts_corr:.3f}")
        if task == "etpc" and etpc_accuracy is not None:
            print(f"Paraphrase Type detection accuracy: {etpc_accuracy:.3f}")

    model.train()

    return (
        quora_accuracy,
        quora_f1,
        quora_y_pred,
        quora_sent_ids,
        sst_accuracy,
        sst_y_pred,
        sst_sent_ids,
        sts_corr,
        sts_y_pred,
        sts_sent_ids,
        etpc_accuracy,
        etpc_y_pred,
        etpc_sent_ids,
    )


def model_eval_test_multitask(
    sst_dataloader, quora_dataloader, sts_dataloader, etpc_dataloader, model, device, task
):
    """
    Evaluate a multitask model on test splits (labels not required).

    Produces predictions and aligned sentence ids for each enabled task.

    Args:
        sst_dataloader (Optional[DataLoader]): SST test loader.
        quora_dataloader (Optional[DataLoader]): QQP test loader.
        sts_dataloader (Optional[DataLoader]): STS test loader.
        etpc_dataloader (Optional[DataLoader]): ETPC test loader.
        model (torch.nn.Module): Multitask model with task heads.
        device (torch.device): Inference device.
        task (str): One of {"sst","sts","qqp","etpc","multitask"}.

    Returns:
        Tuple:
            (
                quora_y_pred (List[int]),
                quora_sent_ids (List[int]),
                sst_y_pred (List[int]),
                sst_sent_ids (List[int]),
                sts_y_pred (List[float]),
                sts_sent_ids (List[int]),
                etpc_y_pred (List[List[int]]),
                etpc_sent_ids (List[int]),
            )
    """
    model.eval()

    with torch.no_grad():
        quora_y_pred, quora_sent_ids = [], []
        running_idx = 0
        if task in ("qqp", "multitask") and quora_dataloader is not None:
            for batch in tqdm(quora_dataloader, desc="eval", disable=TQDM_DISABLE):
                ids1 = batch["token_ids_1"].to(device)
                m1 = batch["attention_mask_1"].to(device)
                ids2 = batch["token_ids_2"].to(device)
                m2 = batch["attention_mask_2"].to(device)
                sent_ids = batch.get("sent_ids", None)

                logits = model.predict_paraphrase(ids1, m1, ids2, m2)
                y_hat = logits.sigmoid().round().flatten().cpu().numpy()

                quora_y_pred.extend(y_hat.tolist())
                b_ids = _ensure_ids(sent_ids, ids1.size(0), running_idx)
                quora_sent_ids.extend(b_ids)
                running_idx += ids1.size(0)

        sts_y_pred, sts_sent_ids = [], []
        running_idx = 0
        if task in ("sts", "multitask") and sts_dataloader is not None:
            for batch in tqdm(sts_dataloader, desc="eval", disable=TQDM_DISABLE):
                ids1 = batch["token_ids_1"].to(device)
                m1 = batch["attention_mask_1"].to(device)
                ids2 = batch["token_ids_2"].to(device)
                m2 = batch["attention_mask_2"].to(device)
                sent_ids = batch.get("sent_ids", None)
                print(sent_ids)
                scores = model.predict_similarity(ids1, m1, ids2, m2).detach().cpu().numpy()
                y_hat = scores.flatten()

                sts_y_pred.extend(y_hat.tolist())
                b_ids = _ensure_ids(sent_ids, ids1.size(0), running_idx)
                sts_sent_ids.extend(b_ids)
                running_idx += ids1.size(0)

        sst_y_pred, sst_sent_ids = [], []
        running_idx = 0
        if task in ("sst", "multitask") and sst_dataloader is not None:
            for batch in tqdm(sst_dataloader, desc="eval", disable=TQDM_DISABLE):
                ids = batch["token_ids"].to(device)
                m = batch["attention_mask"].to(device)
                sent_ids = batch.get("sent_ids", None)

                logits = model.predict_sentiment(ids, m)
                y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

                sst_y_pred.extend(y_hat.tolist())
                b_ids = _ensure_ids(sent_ids, ids.size(0), running_idx)
                sst_sent_ids.extend(b_ids)
                running_idx += ids.size(0)

        etpc_y_pred, etpc_sent_ids = [], []
        running_idx = 0
        if task == "etpc" and etpc_dataloader is not None:
            for batch in tqdm(etpc_dataloader, desc="eval", disable=TQDM_DISABLE):
                ids1 = batch["token_ids_1"].to(device)
                m1 = batch["attention_mask_1"].to(device)
                ids2 = batch["token_ids_2"].to(device)
                m2 = batch["attention_mask_2"].to(device)
                sent_ids = batch.get("sent_ids", None)

                logits = model.predict_paraphrase_types(ids1, m1, ids2, m2)
                y_hat = logits.sigmoid().round().to(torch.int).cpu().tolist()

                etpc_y_pred.extend(y_hat)
                b_ids = _ensure_ids(sent_ids, ids1.size(0), running_idx)
                etpc_sent_ids.extend(b_ids)
                running_idx += ids1.size(0)

        return (
            quora_y_pred,
            quora_sent_ids,
            sst_y_pred,
            sst_sent_ids,
            sts_y_pred,
            sts_sent_ids,
            etpc_y_pred,
            etpc_sent_ids,
        )


def test_model_multitask(args, model, device):
    """
    Run dev and test evaluation for the selected task(s) and write predictions.

    Args:
        args (argparse.Namespace): Contains paths, batch size, and output paths.
        model (torch.nn.Module): Multitask model with task heads.
        device (torch.device): Inference device.

    Returns:
        Tuple or None: Mirrors the return of `model_eval_test_multitask` for non-STS tasks,
        otherwise None after writing files for each enabled task.
    """
    sst_test_raw, _, quora_test_raw, sts_test_raw, etpc_test_raw = load_multitask_data(
        args.sst_test, args.quora_test, args.sts_test, args.etpc_test, split="test"
    )
    sst_dev_raw, _, quora_dev_raw, sts_dev_raw, etpc_dev_raw = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="dev"
    )

    sst_test_ds = SentenceClassificationDataset(sst_test_raw, args)
    sst_dev_ds = SentenceClassificationDataset(sst_dev_raw, args)
    sst_test_dl = DataLoader(sst_test_ds, shuffle=False, batch_size=args.batch_size, collate_fn=sst_test_ds.collate_fn)
    sst_dev_dl = DataLoader(sst_dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_ds.collate_fn)

    quora_test_ds = SentencePairDataset(quora_test_raw, args)
    quora_dev_ds = SentencePairDataset(quora_dev_raw, args)
    quora_test_dl = DataLoader(quora_test_ds, shuffle=False, batch_size=args.batch_size, collate_fn=quora_test_ds.collate_fn)
    quora_dev_dl = DataLoader(quora_dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=quora_dev_ds.collate_fn)

    sts_test_ds = SentencePairDataset(sts_test_raw, args)
    sts_dev_ds = SentencePairDataset(sts_dev_raw, args)
    sts_test_dl = DataLoader(sts_test_ds, shuffle=False, batch_size=args.batch_size, collate_fn=sts_test_ds.collate_fn)
    sts_dev_dl = DataLoader(sts_dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_ds.collate_fn)

    etpc_test_ds = SentencePairDataset(etpc_test_raw, args)
    etpc_dev_ds = SentencePairDataset(etpc_dev_raw, args)
    etpc_test_dl = DataLoader(etpc_test_ds, shuffle=False, batch_size=args.batch_size, collate_fn=etpc_test_ds.collate_fn)
    etpc_dev_dl = DataLoader(etpc_dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=etpc_dev_ds.collate_fn)

    task = args.task

    (
        dev_quora_accuracy,
        quora_dev_f1,
        dev_quora_y_pred,
        dev_quora_sent_ids,
        dev_sst_accuracy,
        dev_sst_y_pred,
        dev_sst_sent_ids,
        dev_sts_corr,
        dev_sts_y_pred,
        dev_sts_sent_ids,
        dev_etpc_accuracy,
        dev_etpc_y_pred,
        dev_etpc_sent_ids,
    ) = model_eval_multitask(
        sst_dev_dl, quora_dev_dl, sts_dev_dl, etpc_dev_dl, model, device, task
    )

    (
        test_quora_y_pred,
        test_quora_sent_ids,
        test_sst_y_pred,
        test_sst_sent_ids,
        test_sts_y_pred,
        test_sts_sent_ids,
        test_etpc_y_pred,
        test_etpc_sent_ids,
    ) = model_eval_test_multitask(
        sst_test_dl, quora_test_dl, sts_test_dl, etpc_test_dl, model, device, task
    )

    if task in ("sst", "multitask"):
        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {0.0 if dev_sst_accuracy is None else dev_sst_accuracy:.3f}")
            f.write("id,Predicted_Sentiment\n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p}\t{s}\n")
        with open(args.sst_test_out, "w+") as f:
            f.write("id,Predicted_Sentiment\n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p}\t{s}\n")

    if task in ("qqp", "multitask"):
        with open(args.quora_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {0.0 if dev_quora_accuracy is None else dev_quora_accuracy:.3f}")
            f.write("id,Predicted_Is_Paraphrase\n")
            for p, s in zip(dev_quora_sent_ids, dev_quora_y_pred):
                f.write(f"{p}\t{s}\n")
        with open(args.quora_test_out, "w+") as f:
            f.write("id,Predicted_Is_Paraphrase\n")
            for p, s in zip(test_quora_sent_ids, test_quora_y_pred):
                f.write(f"{p}\t{s}\n")

    if task in ("sts", "multitask"):
        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts Pearson :: {0.0 if dev_sts_corr is None else dev_sts_corr:.3f}")
            f.write("id,Predicted_Similarity\n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p}\t{s}\n")
        with open(args.sts_test_out, "w+") as f:
            f.write("id,Predicted_Similarity\n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p}\t{s}\n")

    if task == "etpc":
        with open(args.etpc_dev_out, "w+") as f:
            print(f"dev etpc acc :: {0.0 if dev_etpc_accuracy is None else dev_etpc_accuracy:.3f}")
            f.write("id,Predicted_Paraphrase_Types\n")
            for p, s in zip(dev_etpc_sent_ids, dev_etpc_y_pred):
                f.write(f"{p}\t{s}\n")
        with open(args.etpc_test_out, "w+") as f:
            f.write("id,Predicted_Paraphrase_Types\n")
            for p, s in zip(test_etpc_sent_ids, test_etpc_y_pred):
                f.write(f"{p}\t{s}\n")
