import argparse
import os
from logging import disable
from pprint import pformat
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bertSTS import BertModel
from convBert import BertModel as convBertModel  # noqa: F401
from simBert import BertModel as simBertModel

from datasetsSTS import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from buildSimBertFromHF import build_simbert_from_pretrained
from evaluation import model_eval_multitask, test_model_multitask

from transformers import AdamW as HFAdamW
from transformers import get_linear_schedule_with_warmup
from transformers import ConvBertModel as HFConvBertModel

from pcgradOptimizer import PCGrad
from gradvacOptimizer import GradVac

TQDM_DISABLE = False


def seed_everything(seed=11711):
    """
    Set random seeds across Python, NumPy, and PyTorch to improve reproducibility.

    Args:
        seed (int): Seed used for Python, NumPy, and PyTorch RNGs.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


def _spearmanr_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a tie-aware Spearman rank correlation without SciPy.

    Args:
        x (np.ndarray): First sequence of scores.
        y (np.ndarray): Second sequence of scores.

    Returns:
        float: Spearman's rho between x and y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def rankdata(a):
        """
        Assign average ranks to values in a, handling ties.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Ranks with tie-handling via averaging.
        """
        sorter = np.argsort(a, kind="mergesort")
        inv = np.empty_like(sorter)
        inv[sorter] = np.arange(len(a))
        a_sorted = a[sorter]
        obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
        dense_rank = np.cumsum(obs)
        counts = np.bincount(dense_rank)
        csum = np.cumsum(counts)
        ranks = np.empty_like(dense_rank, dtype=float)
        for r in range(1, dense_rank.max() + 1):
            lo = 0 if r == 1 else csum[r - 2]
            hi = csum[r - 1]
            ranks[dense_rank == r] = (lo + hi - 1) / 2.0 + 1.0
        return ranks[inv]

    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


class MultitaskBERT(nn.Module):
    """
    Multitask model with a Transformer backbone and task-specific heads.

    Tasks:
        - Sentiment classification (SST)
        - Paraphrase detection (QQP)
        - Semantic textual similarity (STS)
        - Paraphrase types (ETPC)

    Args:
        config (SimpleNamespace): Model configuration (dropout, hidden size, etc.).
        args (argparse.Namespace): CLI args containing model/backbone selection.

    Attributes:
        bert (nn.Module): The loaded Transformer backbone.
        dropout (nn.Dropout): Dropout module applied before task heads.
        sentiment_classifier (nn.Linear): Classification head for SST.
        sts_regressor (nn.Linear): Regression head for STS.
        paraphrase_classifier (nn.Linear): Binary classifier for QQP.
        paraphrase_types_classifier (nn.Linear): Multi-label head for ETPC.
    """

    def __init__(self, config, args):
        super(MultitaskBERT, self).__init__()

        if args.model == 'bert':
            hf_name = args.hf_model_name or "bert-base-uncased"
            self.bert = BertModel.from_pretrained(hf_name, local_files_only=config.local_files_only)
        elif args.model == 'simBert':
            hf_name = args.hf_model_name or "bert-base-uncased"
            self.bert = build_simbert_from_pretrained(hf_name, local_files_only=config.local_files_only)
        elif args.model == 'convBert':
            hf_name = args.hf_model_name or "YituTech/conv-bert-base"
            self.bert = HFConvBertModel.from_pretrained(hf_name, local_files_only=config.local_files_only)

        name_or_path = getattr(self.bert, "name_or_path", None) or getattr(self.bert, "_name_or_path", None)
        print(f"[Backbone] loaded from: {name_or_path}")

        for p in self.bert.parameters():
            p.requires_grad = (config.option == "finetune")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_classifier = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        self.sts_regressor = nn.Linear(self.bert.config.hidden_size * 2, 1)
        self.paraphrase_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)
        self.paraphrase_types_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, 14)

    def _encode(self, input_ids, attention_mask):
        """
        Encode input sequences and produce attention-masked mean-pooled embeddings.

        Args:
            input_ids (torch.LongTensor): Token IDs of shape [B, T].
            attention_mask (torch.LongTensor): Attention mask of shape [B, T].

        Returns:
            torch.FloatTensor: Mean-pooled embeddings of shape [B, H].
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        return summed / denom

    def forward(self, input_ids, attention_mask):
        """
        Forward pass producing sentence embeddings.

        Args:
            input_ids (torch.LongTensor): Token IDs of shape [B, T].
            attention_mask (torch.LongTensor): Attention mask of shape [B, T].

        Returns:
            torch.FloatTensor: Embeddings of shape [B, H].
        """
        return self._encode(input_ids, attention_mask)

    def predict_sentiment(self, input_ids, attention_mask):
        """
        Produce sentiment logits for SST.

        Args:
            input_ids (torch.LongTensor): Token IDs of shape [B, T].
            attention_mask (torch.LongTensor): Attention mask of shape [B, T].

        Returns:
            torch.FloatTensor: Class logits of shape [B, 5].
        """
        cls = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(self.dropout(cls))
        return logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Predict paraphrase logits for QQP.

        Args:
            input_ids_1 (torch.LongTensor): First sentence IDs [B, T].
            attention_mask_1 (torch.LongTensor): First sentence mask [B, T].
            input_ids_2 (torch.LongTensor): Second sentence IDs [B, T].
            attention_mask_2 (torch.LongTensor): Second sentence mask [B, T].

        Returns:
            torch.FloatTensor: Logits of shape [B], compatible with BCEWithLogits.
        """
        emb1 = self.forward(input_ids_1, attention_mask_1)
        emb2 = self.forward(input_ids_2, attention_mask_2)
        combined = torch.cat((emb1, emb2), dim=1)
        logit = self.paraphrase_classifier(self.dropout(combined))
        return logit.squeeze(-1)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Regress a semantic similarity score for STS.

        Args:
            input_ids_1 (torch.LongTensor): First sentence IDs [B, T].
            attention_mask_1 (torch.LongTensor): First sentence mask [B, T].
            input_ids_2 (torch.LongTensor): Second sentence IDs [B, T].
            attention_mask_2 (torch.LongTensor): Second sentence mask [B, T].

        Returns:
            torch.FloatTensor: Unbounded scores of shape [B] (use MSE vs. gold 0..5).
        """
        cls1 = self.forward(input_ids_1, attention_mask_1)
        cls2 = self.forward(input_ids_2, attention_mask_2)
        combined = torch.cat([cls1, cls2], dim=1)
        score = self.sts_regressor(self.dropout(combined)).squeeze(-1)
        return score

    def predict_paraphrase_types(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Predict multi-label paraphrase types for ETPC.

        Args:
            input_ids_1 (torch.LongTensor): First sentence IDs [B, T].
            attention_mask_1 (torch.LongTensor): First sentence mask [B, T].
            input_ids_2 (torch.LongTensor): Second sentence IDs [B, T].
            attention_mask_2 (torch.LongTensor): Second sentence mask [B, T].

        Returns:
            torch.FloatTensor: Logits of shape [B, 14], suitable for BCEWithLogits.
        """
        emb1 = self.forward(input_ids_1, attention_mask_1)
        emb2 = self.forward(input_ids_2, attention_mask_2)
        combined = torch.cat((emb1, emb2), dim=1)
        logits = self.paraphrase_types_classifier(self.dropout(combined))
        return logits


def save_model(model, optimizer, args, config, filepath):
    """
    Save model, optimizer, and RNG states to a checkpoint.

    Args:
        model (nn.Module): Trained model to serialize.
        optimizer (torch.optim.Optimizer): Optimizer whose state will be saved.
        args (argparse.Namespace): Full CLI arguments used for training.
        config (SimpleNamespace): Model config used to build the model.
        filepath (str): Destination path for the checkpoint file.

    Returns:
        None
    """
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(save_info, filepath)
    print(f"Saving the model to {filepath}.")


def _build_dataloaders(args):
    """
    Construct task-specific datasets and dataloaders from CSV files.

    Args:
        args (argparse.Namespace): CLI args containing paths, task, and batch size.

    Returns:
        tuple: Dataloaders in the order
            (sst_train, sst_dev, quora_train, quora_dev, sts_train, sts_dev, etpc_train, etpc_dev),
            where individual entries may be None if not requested by --task.
    """
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="dev"
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None

    if args.task in ("sst", "multitask"):
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
        sst_train_dataloader = DataLoader(
            sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn
        )

    if args.task in ("sts", "multitask"):
        sts_train_data = SentencePairDataset(sts_train_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args)
        sts_train_dataloader = DataLoader(
            sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn
        )

    if args.task in ("qqp", "multitask"):
        quora_train_data = SentencePairDataset(quora_train_data, args)
        quora_dev_data = SentencePairDataset(quora_dev_data, args)
        quora_train_dataloader = DataLoader(
            quora_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=quora_train_data.collate_fn
        )
        quora_dev_dataloader = DataLoader(
            quora_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=quora_dev_data.collate_fn
        )

    if args.task in ("etpc", "multitask"):
        etpc_train_data = SentencePairDataset(etpc_train_data, args)
        etpc_dev_data = SentencePairDataset(etpc_dev_data, args)
        etpc_train_dataloader = DataLoader(
            etpc_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=etpc_train_data.collate_fn
        )
        etpc_dev_dataloader = DataLoader(
            etpc_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=etpc_dev_data.collate_fn
        )

    return (
        sst_train_dataloader,
        sst_dev_dataloader,
        quora_train_dataloader,
        quora_dev_dataloader,
        sts_train_dataloader,
        sts_dev_dataloader,
        etpc_train_dataloader,
        etpc_dev_dataloader,
    )


@torch.no_grad()
def _eval_sts_pearson(dataloader, model, device, save_path=None):
    """
    Evaluate STS with Pearson r and MSE. Optionally write predictions in the
    'id,Predicted_Similarity' format where IDs come from the input CSV.

    Args:
        dataloader: STS dataloader that yields batches with keys:
            token_ids_1, attention_mask_1, token_ids_2, attention_mask_2,
            labels (optional on test), sent_ids (list of str or tensor)
        model: model with predict_similarity(...)
        device: torch.device
        save_path (str | None): if provided, write 'id,Predicted_Similarity' file.

    Returns:
        (pearson_r: float, mse: float)
    """
    model.eval()
    preds, golds, all_ids = [], [], []
    print(save_path)
    for batch in tqdm(dataloader, desc="eval", disable=TQDM_DISABLE):
        ids1 = batch["token_ids_1"].to(device)
        m1   = batch["attention_mask_1"].to(device)
        ids2 = batch["token_ids_2"].to(device)
        m2   = batch["attention_mask_2"].to(device)

        # sent_ids can be a list of strings (preferred) or a tensor of ints
        batch_ids = batch.get("sent_ids", None)
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.detach().cpu().tolist()
        # ensure list of strings
        batch_ids = [str(x) for x in (batch_ids or [])]
        all_ids.extend(batch_ids)

        out = model.predict_similarity(ids1, m1, ids2, m2).detach().cpu().numpy()
        preds.append(out)

        if "labels" in batch and batch["labels"] is not None:
            golds.append(batch["labels"].detach().cpu().numpy())

    if len(preds) == 0:
        # Nothing to evaluate
        if save_path:
            # still write an empty header to avoid downstream surprises
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("id,Predicted_Similarity\n")
        return 0.0, 0.0

    preds = np.concatenate(preds, axis=0)
    preds_rep = np.clip(preds, 0.0, 5.0)

    if golds:
        golds = np.concatenate(golds, axis=0)
        pearson_mat = np.corrcoef(preds_rep, golds)
        r = float(pearson_mat[1][0])
        mse = float(np.mean((preds_rep - golds) ** 2))
    else:
        # test split without golds
        r = float("nan")
        mse = float("nan")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Align lengths defensively in case something odd happened
        n = min(len(all_ids), len(preds_rep))
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("id,Predicted_Similarity\n")
            for sid, score in zip(all_ids[:n], preds_rep[:n]):
                # Header uses comma per spec; body uses tab between id and value
                f.write(f"{sid}\t{float(score)}\n")

    return r, mse

def train_multitask(args):
    """
    Train the selected task(s) and evaluate after each epoch.

    Args:
        args (argparse.Namespace): Training configuration, paths, and hyperparameters.

    Returns:
        None
    """
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    (
        sst_train_dataloader,
        sst_dev_dataloader,
        quora_train_dataloader,
        quora_dev_dataloader,
        sts_train_dataloader,
        sts_dev_dataloader,
        etpc_train_dataloader,
        etpc_dev_dataloader,
    ) = _build_dataloaders(args)

    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
    }
    config = SimpleNamespace(**config)

    print("-" * 30)
    print("    BERT/ConvBERT Model Configuration")
    print("-" * 30)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print("-" * 30)

    if args.task in ("sts", "multitask") and args.option == "pretrain":
        print("[Warn] STS with frozen encoder usually underperforms. Consider --option finetune.")

    model = MultitaskBERT(config, args).to(device)

    use_special_optimizer = args.optimizer in ("pcgrad", "gradvac")

    if not use_special_optimizer:
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        grouped = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = HFAdamW(grouped, lr=args.lr)

        steps_per_epoch = 0
        for dl in (sst_train_dataloader, quora_train_dataloader, sts_train_dataloader, etpc_train_dataloader):
            if dl is not None:
                steps_per_epoch += len(dl)
        total_steps = max(1, steps_per_epoch * args.epochs)
        warmup_steps = int(total_steps * args.warmup_ratio)
        print(f"[Scheduler] steps/epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    else:
        optimizer = HFAdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    best_dev_acc = float("-inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        def _opt_step(loss):
            """
            Apply backward, gradient clipping, and optimizer/scheduler steps.

            Args:
                loss (torch.Tensor): Scalar loss to backpropagate.

            Returns:
                None
            """
            nonlocal train_loss, num_batches
            if use_special_optimizer:
                if args.optimizer == "pcgrad":
                    pcg = PCGrad(optimizer)
                    pcg.pc_backward(loss)
                elif args.optimizer == "gradvac":
                    gv = GradVac(optimizer, reduction="sum", target=0.0, alpha=0.5)
                    gv.gv_backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            train_loss += loss.item()
            num_batches += 1

        if args.task in ("sst", "multitask"):
            for batch in tqdm(sst_train_dataloader, desc=f"train-{epoch + 1:02}-sst", disable=TQDM_DISABLE):
                b_ids = batch["token_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                b_labels = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(not use_special_optimizer) and scaler.is_enabled()):
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1))
                if not use_special_optimizer:
                    scaler.scale(loss).backward()
                _opt_step(loss)

        if args.task in ("sts", "multitask"):
            for batch in tqdm(sts_train_dataloader, desc=f"train-{epoch + 1:02}-sts", disable=TQDM_DISABLE):
                ids1 = batch["token_ids_1"].to(device)
                mask1 = batch["attention_mask_1"].to(device)
                ids2 = batch["token_ids_2"].to(device)
                mask2 = batch["attention_mask_2"].to(device)
                labels = batch["labels"].to(device).float()
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(not use_special_optimizer) and scaler.is_enabled()):
                    preds = model.predict_similarity(ids1, mask1, ids2, mask2)
                    loss = F.mse_loss(preds, labels)
                if not use_special_optimizer:
                    scaler.scale(loss).backward()
                _opt_step(loss)

        if args.task in ("qqp", "multitask"):
            for batch in tqdm(quora_train_dataloader, desc=f"train-{epoch + 1:02}-qqp", disable=TQDM_DISABLE):
                ids1 = batch["token_ids_1"].to(device)
                mask1 = batch["attention_mask_1"].to(device)
                ids2 = batch["token_ids_2"].to(device)
                mask2 = batch["attention_mask_2"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(not use_special_optimizer) and scaler.is_enabled()):
                    logits = model.predict_paraphrase(ids1, mask1, ids2, mask2)
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float().view(-1))
                if not use_special_optimizer:
                    scaler.scale(loss).backward()
                _opt_step(loss)

        if args.task in ("etpc", "multitask"):
            for batch in tqdm(etpc_train_dataloader, desc=f"train-{epoch + 1:02}-etpc", disable=TQDM_DISABLE):
                ids1 = batch["token_ids_1"].to(device)
                mask1 = batch["attention_mask_1"].to(device)
                ids2 = batch["token_ids_2"].to(device)
                mask2 = batch["attention_mask_2"].to(device)
                labels = batch["labels"].to(device).float()
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(not use_special_optimizer) and scaler.is_enabled()):
                    logits = model.predict_paraphrase_types(ids1, mask1, ids2, mask2)
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                if not use_special_optimizer:
                    scaler.scale(loss).backward()
                _opt_step(loss)

        train_loss = train_loss / max(1, num_batches)

        if args.task == "sts":
            sts_train_r, sts_train_mse = _eval_sts_pearson(sts_train_dataloader, model, device)
            sts_dev_r, sts_dev_mse = _eval_sts_pearson(sts_dev_dataloader, model, device, save_path=args.sts_dev_out)
            train_acc, dev_acc = sts_train_r, sts_dev_r
            print(f"Epoch {epoch + 1:02} (sts): DEV Pearson {sts_dev_r:.4f} | DEV MSE {sts_dev_mse:.4f}")
        else:
            _ = model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
            _ = model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
            train_acc, dev_acc = 0.0, 0.0

        print(
            f"Epoch {epoch + 1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        if args.optimizer in ("pcgrad", "gradvac") and args.task != "multitask":
            print(f"The Optimizer {args.optimizer} is only for multitasks.")
            sys.exit(1)


def test_model(args):
    """
    Load a checkpoint and run test-time evaluation.

    Args:
        args (argparse.Namespace): CLI args including checkpoint path and task.

    Returns:
        Any: The return of the underlying evaluation helper for non-STS tasks.
    """
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    saved = torch.load(args.filepath, map_location=device)
    config = saved["model_config"]

    model = MultitaskBERT(config, args).to(device)
    model.load_state_dict(saved["model"])
    print(f"Loaded model to test from {args.filepath}")

    if args.task == "sts":
        _, _, _, sts_dev_raw, _ = load_multitask_data(
            args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="dev"
        )
        _, _, _, sts_test_raw, _ = load_multitask_data(
            args.sst_test, args.quora_test, args.sts_test, args.etpc_test, split="test"
        )

        sts_dev_ds = SentencePairDataset(sts_dev_raw, args)
        sts_test_ds = SentencePairDataset(sts_test_raw, args)
        sts_dev_loader = DataLoader(
            sts_dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_ds.collate_fn
        )
        sts_test_loader = DataLoader(
            sts_test_ds, shuffle=False, batch_size=args.batch_size, collate_fn=sts_test_ds.collate_fn
        )

        dev_r, dev_mse = _eval_sts_pearson(sts_dev_loader, model, device, save_path=args.sts_dev_out)
        test_r, test_mse = _eval_sts_pearson(sts_test_loader, model, device, save_path=args.sts_test_out)

        print(f"[TEST] STS DEV  Pearson: {dev_r:.4f} | MSE: {dev_mse:.4f} | saved: {args.sts_dev_out}")
        return

    return test_model_multitask(args, model, device)


def get_args():
    """
    Parse and return command-line arguments for training and evaluation.

    Returns:
        argparse.Namespace: Parsed arguments with defaults suitable for STS + ConvBERT.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", type=str, choices=("sst", "sts", "qqp", "etpc", "multitask"), default="sts"
    )
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--option",
        type=str,
        choices=("pretrain", "finetune"),
        default="finetune",
        help="When 'finetune', encoder parameters are updated.",
    )
    parser.add_argument("--optimizer", type=str, choices=["pcgrad", "gradvac"], default="")
    parser.add_argument("--model", type=str, choices=("bert", "convBert", "simBert"), default="convBert")
    parser.add_argument("--hf_model_name", type=str, default="YituTech/conv-bert-base")
    parser.add_argument("--use_gpu", action="store_true")

    args, _ = parser.parse_known_args()

    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/sst-sentiment-test-student.csv")

    parser.add_argument("--quora_train", type=str, default="data/quora-paraphrase-train.csv")
    parser.add_argument("--quora_dev", type=str, default="data/quora-paraphrase-dev.csv")
    parser.add_argument("--quora_test", type=str, default="data/quora-paraphrase-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-similarity-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-similarity-test-student.csv")

    parser.add_argument("--etpc_train", type=str, default="data/etpc-paraphrase-train.csv")
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-generation-test-student.csv")
    parser.add_argument("--etpc_test", type=str, default="data/etpc-paraphrase-detection-test-student.csv")

    parser.add_argument("--sst_dev_out", type=str, default="predictions/bert/sst-sentiment-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/bert/sst-sentiment-test-output.csv")
    parser.add_argument("--quora_dev_out", type=str, default="predictions/bert/quora-paraphrase-dev-output.csv")
    parser.add_argument("--quora_test_out", type=str, default="predictions/bert/quora-paraphrase-test-output.csv")
    parser.add_argument("--sts_dev_out", type=str, default="predictions/bert/sts-similarity-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/bert/sts-similarity-test-student.csv")
    parser.add_argument("--etpc_dev_out", type=str, default="predictions/bert/etpc-paraphrase-detection-test-student.csv")
    parser.add_argument("--etpc_test_out", type=str, default="predictions/bert/etpc-paraphrase-generation-test-student.csv")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision when CUDA is available.")
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"
    seed_everything(args.seed)
    train_multitask(args)
    test_model(args)
