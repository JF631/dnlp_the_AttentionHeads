"""
This script implements the same multitask training workflow as `multitask_classifier.py`,
but **excludes** the ETPC dataset when the `--task multitask` option is selected.

In other words:
- Single-task modes (`--task sst`, `--task sts`, `--task qqp`, `--task etpc`) behave as usual.
- Multitask mode jointly trains on SST, QQP, and STS **only**; ETPC is intentionally omitted.
"""
import argparse
from pprint import pformat
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from convBert import BertModel as convBertModel
from simBert import BertModel as simBertModel
from datasetsSTS import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from buildSimBertFromHF import build_simbert_from_pretrained
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW
from pcgradOptimizer import PCGrad
from gradvacOptimizer import GradVac

TQDM_DISABLE = False

def seed_everything(seed=11711):
    """
    Set random seeds across Python, NumPy, and PyTorch to improve reproducibility.

    Notes:
        - `torch.backends.cudnn.deterministic = True` trades some speed for determinism.
        - Perfect reproducibility is not guaranteed across different hardware/driver/cuDNN
          versions or when using certain nondeterministic ops.

    Parameters:
        seed (int): The base random seed applied to Python, NumPy, and PyTorch RNGs.
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

class MultitaskBERT(nn.Module):
    """
    BERT-based model with task-specific heads for:
      - Sentiment classification (SST) via `predict_sentiment`
      - Paraphrase detection (QQP) via `predict_paraphrase`
      - Semantic Textual Similarity (STS) via `predict_similarity`
      - ETPC paraphrase type detection (multi-label) via `predict_paraphrase_types`

    Important:
        In `multitask` mode, training includes SST, QQP, and STS only.
        ETPC heads exist for single-task `--task etpc` usage.

    Attributes:
        bert (nn.Module): Backbone encoder (`bert`, `simBert`, or `convBert`).
        dropout (nn.Dropout): Applied before task heads.
        sentiment_classifier (nn.Linear): Maps [B, H] -> [B, 5] for SST.
        sts_regressor (nn.Linear): Maps [B, 2H] -> [B, 1] for STS (score in [0, 5]).
        paraphrase_classifier (nn.Linear): Maps [B, 2H] -> [B, 1] for QQP logits.
        paraphrase_types_classifier (nn.Linear): Maps [B, 2H] -> [B, 14] for ETPC logits.

    Config options:
        - `option`: "pretrain" (freeze backbone) or "finetune" (update backbone)
        - `hidden_dropout_prob`: Dropout probability applied to pooled embeddings
        - `local_files_only`: If True, do not fetch weights from the internet
        - `hidden_size`: Expected BERT hidden size (default 768)
    """
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        # Select the backbone encoder based on args.model
        if args.model == 'bert':
            self.bert = BertModel.from_pretrained(
                "bert-base-uncased", local_files_only=config.local_files_only
            )
        elif args.model == 'simBert':
            self.bert = build_simbert_from_pretrained(
                "bert-base-uncased", local_files_only=config.local_files_only
            )
        elif args.model == 'convBert':
            self.bert = convBertModel.from_pretrained(
                "bert-base-uncased", local_files_only=config.local_files_only
            )

        # Freeze or unfreeze the backbone depending on training option
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        # General dropout layer using hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Task heads
        self.sentiment_classifier = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)

        # STS uses concatenated sentence embeddings: shape [B, 2H] -> [B, 1]
        self.sts_regressor = nn.Linear(self.bert.config.hidden_size * 2, 1)

        # QQP uses concatenated embeddings: shape [B, 2H] -> [B, 1] (logit)
        self.paraphrase_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)

        # ETPC paraphrase type detection (multi-label). 14 binary labels -> logits.
        self.paraphrase_types_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, 14)

    def forward(self, input_ids, attention_mask):
        """
        Encode a batch of sentences and return the pooled [CLS] embeddings.

        Parameters:
            input_ids (torch.LongTensor): Token ids, shape [B, T].
            attention_mask (torch.LongTensor): Attention mask, shape [B, T].

        Returns:
            torch.FloatTensor: Pooled embeddings from the backbone encoder,
            shape [B, H]. (Accessed via `output['pooler_output']`.)
        """
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return bert_output['pooler_output']

    def predict_sentiment(self, input_ids, attention_mask):
        """
        Predict sentiment class logits for SST.

        Parameters:
            input_ids (torch.LongTensor): Token ids, shape [B, T].
            attention_mask (torch.LongTensor): Attention mask, shape [B, T].

        Returns:
            torch.FloatTensor: Unnormalized class scores (logits), shape [B, 5].
            Use `F.cross_entropy(logits, labels)` during training and `argmax(-1)` for labels.
        """
        cls_embedding = self.forward(input_ids, attention_mask)           # [B, H]
        logits_after_dropout = self.dropout(cls_embedding)                # [B, H]
        logits = self.sentiment_classifier(logits_after_dropout)          # [B, 5]
        return logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Predict paraphrase logits for QQP using concatenated sentence embeddings.

        Parameters:
            input_ids_1 (torch.LongTensor): First sentence ids, shape [B, T].
            attention_mask_1 (torch.LongTensor): First sentence mask, shape [B, T].
            input_ids_2 (torch.LongTensor): Second sentence ids, shape [B, T].
            attention_mask_2 (torch.LongTensor): Second sentence mask, shape [B, T].

        Returns:
            torch.FloatTensor: Logits for the positive "is paraphrase" class, shape [B].
            Apply `torch.sigmoid(logits)` to get probabilities in [0, 1].
        """
        emb1 = self.forward(input_ids_1, attention_mask_1)                # [B, H]
        emb2 = self.forward(input_ids_2, attention_mask_2)                # [B, H]
        combined_emb = torch.cat((emb1, emb2), dim=1)                     # [B, 2H]
        dropped_emb = self.dropout(combined_emb)                          # [B, 2H]
        logits = self.paraphrase_classifier(dropped_emb)                  # [B, 1]
        return logits.squeeze(-1)                                         # [B]

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Regress a semantic similarity score for STS on a 0–5 scale.

        Parameters:
            input_ids_1 (torch.LongTensor): First sentence ids, shape [B, T].
            attention_mask_1 (torch.LongTensor): First sentence mask, shape [B, T].
            input_ids_2 (torch.LongTensor): Second sentence ids, shape [B, T].
            attention_mask_2 (torch.LongTensor): Second sentence mask, shape [B, T].

        Returns:
            torch.FloatTensor: Similarity scores scaled to [0, 5], shape [B].
            Uses a sigmoid on a linear head and rescales by 5.
        """
        cls_1 = self.forward(input_ids_1, attention_mask_1)               # [B, H]
        cls_2 = self.forward(input_ids_2, attention_mask_2)               # [B, H]
        combined = torch.cat([cls_1, cls_2], dim=1)                       # [B, 2H]
        combined = self.dropout(combined)
        similarity = self.sts_regressor(combined)                         # [B, 1]
        similarity = torch.sigmoid(similarity) * 5                        # -> [0, 5]
        return similarity.view(-1)                                        # [B]

    def predict_paraphrase_types(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Predict ETPC paraphrase types as a 14-dimensional multi-label classification.

        Parameters:
            input_ids_1 (torch.LongTensor): First sentence ids, shape [B, T].
            attention_mask_1 (torch.LongTensor): First sentence mask, shape [B, T].
            input_ids_2 (torch.LongTensor): Second sentence ids, shape [B, T].
            attention_mask_2 (torch.LongTensor): Second sentence mask, shape [B, T].

        Returns:
            torch.FloatTensor: Raw logits for each of the 14 labels, shape [B, 14].
            Use `F.binary_cross_entropy_with_logits(logits, labels)` for training,
            and `torch.sigmoid(logits) > threshold` for prediction.
        """
        emb1 = self.forward(input_ids_1, attention_mask_1)                # [B, H]
        emb2 = self.forward(input_ids_2, attention_mask_2)                # [B, H]
        combined_emb = torch.cat((emb1, emb2), dim=1)                     # [B, 2H]
        dropped_emb = self.dropout(combined_emb)                          # [B, 2H]
        logits = self.paraphrase_types_classifier(dropped_emb)            # [B, 14]
        return logits


def save_model(model, optimizer, args, config, filepath):
    """
    Serialize training state to a `.pt` file.

    Contents:
        - model (state_dict): Weights of `MultitaskBERT`.
        - optim (state_dict): Optimizer state for exact training resumption.
        - args (argparse.Namespace): CLI arguments used to run training.
        - model_config (SimpleNamespace): Model configuration as used to build the model.
        - system_rng (tuple): Python `random.getstate()` for RNG reproducibility.
        - numpy_rng (tuple): NumPy `np.random.get_state()` for RNG reproducibility.
        - torch_rng (torch.ByteTensor): Torch RNG state from `torch.random.get_rng_state()`.

    Parameters:
        model (nn.Module): The model instance to save.
        optimizer (torch.optim.Optimizer): The optimizer whose state is saved.
        args (argparse.Namespace): Parsed command-line arguments.
        config (SimpleNamespace): Model configuration used to construct the module.
        filepath (str): Destination path for the serialized checkpoint.
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

    torch.save(save_info, filepath)
    print(f"Saving the model to {filepath}.")


def train_multitask(args):
    """
    Train the selected task(s) and evaluate on dev after each epoch.

    Workflow:
        1) Load datasets for SST/QQP/STS/ETPC.
        2) Build DataLoaders based on `--task`. In `multitask`, ETPC is excluded.
        3) Initialize the model and optimizer.
        4) Train for `--epochs`, computing task-appropriate losses.
        5) Evaluate on train/dev splits; track the best dev score.
        6) Save a checkpoint when the dev aggregate improves.

    Aggregation (multitask only):
        - SST and QQP use accuracy in [0, 1].
        - STS uses Spearman correlation in [-1, 1], mapped to [0, 1] as (rho+1)/2.
        - ETPC is excluded from the multitask aggregate by design.

    Parameters:
        args (argparse.Namespace): Command-line arguments controlling data paths,
            task selection, optimizer choice, training hyperparameters, device, and I/O.

    Notes:
        `--optimizer pcgrad|gradvac` is intended for true multi-task gradients.
        A safety check errors out if those are requested in single-task mode.
    """
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load raw data for all tasks (we will wrap into datasets conditionally)
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="dev"
    )

    # Initialize dataloader placeholders (set per task below)
    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None

    # SST (single-task or multitask)
    if args.task == "sst" or args.task == "multitask":
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_train_dataloader = DataLoader(
            sst_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sst_train_data.collate_fn,
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sst_dev_data.collate_fn,
        )

    # STS (single-task or multitask)
    if args.task in ("sts", "multitask"):
        sts_train_data = SentencePairDataset(sts_train_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args)

        sts_train_dataloader = DataLoader(
            sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn
        )

    # QQP (single-task or multitask)
    if args.task == "qqp" or args.task == "multitask":
        quora_train_data = SentencePairDataset(quora_train_data, args)
        quora_dev_data = SentencePairDataset(quora_dev_data, args)

        quora_train_dataloader = DataLoader(
            quora_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=quora_train_data.collate_fn
        )
        quora_dev_dataloader = DataLoader(
            quora_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=quora_dev_data.collate_fn
        )

    # ETPC (single-task only; excluded from multitask by design)
    if args.task == "etpc":
        etpc_train_data = SentencePairDataset(etpc_train_data, args)
        etpc_dev_data = SentencePairDataset(etpc_dev_data, args)

        etpc_train_dataloader = DataLoader(
            etpc_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=etpc_train_data.collate_fn,
        )
        etpc_dev_dataloader = DataLoader(
            etpc_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=etpc_dev_data.collate_fn,
        )

    # Build config for the model constructor
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
    }
    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = MultitaskBERT(config).to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    best_dev_acc = float("-inf")
    # Main training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.task == "sst" or args.task == "multitask":
            # Train on SST (cross-entropy over 5 classes)
            for batch in tqdm(
                sst_train_dataloader, desc=f"train-{epoch + 1:02}", disable=TQDM_DISABLE
            ):
                b_ids, b_mask, b_labels = (
                    batch["token_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)
                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1))

                # Multitask optimizers only make sense in multitask mode.
                if args.optimizer == "pcgrad":
                    pcg = PCGrad(optimizer)
                    pcg.pc_backward(loss)
                elif args.optimizer == "gradvac":
                    gv = GradVac(optimizer, reduction="sum", target=0.0, alpha=0.5)
                    gv.gv_backward(loss)
                else:
                    loss.backward()

                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

        if args.task == "sts" or args.task == "multitask":
            # Train on STS (MSE between predicted score in [0,5] and gold)
            for batch in tqdm(sts_train_dataloader, desc=f"train-{epoch + 1:02}-sts", disable=TQDM_DISABLE):
                input_ids_1, attention_mask_1 = batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device)
                input_ids_2, attention_mask_2 = batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad()
                preds = model.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = F.mse_loss(preds, labels.float())

                if args.optimizer == "pcgrad":
                    pcg = PCGrad(optimizer)
                    pcg.pc_backward(loss)
                elif args.optimizer == "gradvac":
                    gv = GradVac(optimizer, reduction="sum", target=0.0, alpha=0.5)
                    gv.gv_backward(loss)
                else:
                    loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

        if args.task == "qqp" or args.task == "multitask":
            # Train on QQP (binary cross-entropy with logits)
            for batch in tqdm(quora_train_dataloader, desc=f"train-{epoch + 1:02}", disable=TQDM_DISABLE):
                # Move batch to device
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                    batch['token_ids_1'].to(device), batch['attention_mask_1'].to(device),
                    batch['token_ids_2'].to(device), batch['attention_mask_2'].to(device),
                    batch['labels'].to(device)
                )

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.float().view(-1))

                if args.optimizer == "pcgrad":
                    pcg = PCGrad(optimizer)
                    pcg.pc_backward(loss)
                elif args.optimizer == "gradvac":
                    gv = GradVac(optimizer, reduction="sum", target=0.0, alpha=0.5)
                    gv.gv_backward(loss)
                else:
                    loss.backward()

                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

        # Train on ETPC only in single-task mode (excluded from multitasking)
        if args.task == "etpc":
            for batch in tqdm(etpc_train_dataloader, desc=f"train-{epoch + 1:02}-etpc", disable=TQDM_DISABLE):
                b_ids1 = batch["token_ids_1"].to(device)
                b_mask1 = batch["attention_mask_1"].to(device)
                b_ids2 = batch["token_ids_2"].to(device)
                b_mask2 = batch["attention_mask_2"].to(device)
                b_labels = batch["labels"].to(device).float()  # shape [B, 14]

                optimizer.zero_grad()
                logits = model.predict_paraphrase_types(b_ids1, b_mask1, b_ids2, b_mask2)  # [B, 14]
                loss = F.binary_cross_entropy_with_logits(logits, b_labels)

                if args.optimizer == "pcgrad":
                    pcg = PCGrad(optimizer)
                    pcg.pc_backward(loss)
                elif args.optimizer == "gradvac":
                    gv = GradVac(optimizer, reduction="sum", target=0.0, alpha=0.5)
                    gv.gv_backward(loss)
                else:
                    loss.backward()

                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / max(num_batches, 1)

        # Compute per-task metrics (train/dev); some may be None depending on task
        quora_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _, etpc_train_acc, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        quora_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, _, _, etpc_dev_acc, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        # Aggregate a single "dev_acc" for model selection
        if args.task == "multitask":
            dev_parts = []
            train_parts = []

            if sst_dev_dataloader is not None:
                dev_parts.append(sst_dev_acc)              # accuracy in [0,1]
                train_parts.append(sst_train_acc)

            if quora_dev_dataloader is not None:
                dev_parts.append(quora_dev_acc)            # accuracy in [0,1]
                train_parts.append(quora_train_acc)

            # ETPC intentionally excluded from multitask (dataloaders are None)

            if sts_dev_dataloader is not None:
                # Map Spearman [-1,1] -> [0,1] for averaging
                dev_parts.append((sts_dev_corr + 1.0) / 2.0)
                train_parts.append((sts_train_corr + 1.0) / 2.0)

            train_acc = sum(train_parts) / len(train_parts) if train_parts else 0.0
            dev_acc = sum(dev_parts) / len(dev_parts) if dev_parts else 0.0

        else:
            # Single-task metrics
            train_acc, dev_acc = {
                "sst": (sst_train_acc, sst_dev_acc),
                "sts": (sts_train_corr, sts_dev_corr),
                "qqp": (quora_train_acc, quora_dev_acc),
                "etpc": (etpc_train_acc, etpc_dev_acc),
            }[args.task]

        print(
            f"Epoch {epoch + 1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        # Checkpoint on dev improvement
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        # Guardrail: multi-task optimizers are not meaningful in single-task setting
        if args.optimizer in ('pcgrad', 'gradvac') and args.task != "multitask":
            print(f"The optimizer '{args.optimizer}' is only supported for --task multitask.")
            exit(1)


def test_model(args):
    """
    Load a saved checkpoint and evaluate on the test splits for the selected task(s).

    Behavior:
        - Restores model architecture from the stored `model_config`.
        - Loads weights from `args.filepath`.
        - Runs task-appropriate test evaluation via `test_model_multitask`.

    Parameters:
        args (argparse.Namespace): Must contain `--filepath` and `--task`.

    Returns:
        Any: Whatever `test_model_multitask` returns for the chosen task(s), typically
        a metrics dictionary or tuple (see `evaluation.py` for specifics).
    """
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        return test_model_multitask(args, model, device)

def get_args():
    """
    Parse command-line arguments controlling data, model, optimizer, and I/O.

    Returns:
        argparse.Namespace:
        The populated argument object. Key groups:

            - Training task: `--task {sst,sts,qqp,etpc,multitask}`
            - Model config: `--option {pretrain,finetune}`, `--model {bert,convBert,simBert}`
            - Optimizer: `--optimizer {pcgrad,gradvac}` (only meaningful for multitask)
            - Paths: dataset CSVs and prediction output paths for each split
            - Hyperparameters: `--epochs`, `--batch_size`, `--lr`, `--hidden_dropout_prob`
            - System: `--seed`, `--use_gpu`, `--local_files_only`
    """
    parser = argparse.ArgumentParser()

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='Choose one of {"sst","sts","qqp","etpc","multitask"} to select the training objective(s).',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
        default="sst",
    )

    # Model configuration
    parser.add_argument("--seed", type=int, default=11711, help="Base RNG seed for Python/NumPy/PyTorch.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--option",
        type=str,
        help="Backbone update option: 'pretrain' freezes BERT; 'finetune' updates BERT parameters.",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )

    # Optimizer Arguments
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Multi-task gradient handling optimizer (use only with --task multitask).",
        choices=["pcgrad", "gradvac"],
        default="",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Backbone encoder to use.",
        choices=("bert", "convBert", "simBert"),  # Either standard BERT, convolutional BERT, or Siamese BERT
        default="bert"
    )

    parser.add_argument("--use_gpu", action="store_true", help="If set, uses CUDA when available.")

    args, _ = parser.parse_known_args()

    # Dataset paths
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/sst-sentiment-test-student.csv")

    parser.add_argument("--quora_train", type=str, default="data/quora-paraphrase-train.csv")
    parser.add_argument("--quora_dev", type=str, default="data/quora-paraphrase-dev.csv")
    parser.add_argument("--quora_test", type=str, default="data/quora-paraphrase-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-similarity-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-similarity-test-student.csv")

    # You should split the train data into a train and dev set first and change the
    # default path of the --etpc_dev argument to your dev set.
    parser.add_argument("--etpc_train", type=str, default="data/etpc-paraphrase-train.csv")
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-generation-test-student.csv")

    parser.add_argument(
        "--etpc_test", type=str, default="data/etpc-paraphrase-detection-test-student.csv"
    )

    # Output paths (default destinations differ in multitask mode)
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-dev-output.csv"
        ),
        help="Output CSV for SST dev predictions."
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-test-output.csv"
        ),
        help="Output CSV for SST test predictions."
    )

    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-dev-output.csv"
        ),
        help="Output CSV for QQP dev predictions."
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-test-output.csv"
        ),
        help="Output CSV for QQP test predictions."
    )

    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
        help="Output CSV for STS dev predictions."
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-student.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-student.csv"
        ),
        help="Output CSV for STS test predictions."
    )

    parser.add_argument(
        "--etpc_dev_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-test-student.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-test-student.csv"
        ),
        help="Output CSV for ETPC dev predictions (single-task only)."
    )
    parser.add_argument(
        "--etpc_test_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-generation-test-student.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-generation-test-student.csv"
        ),
        help="Output CSV for ETPC test predictions (single-task only)."
    )

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Typical: 64 fits a 12GB GPU for SST/QQP/STS.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="Dropout applied before task heads.")
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate. Defaults to 1e-3 for 'pretrain' and 1e-5 for 'finetune'.",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument("--local_files_only", action="store_true", help="Load pretrained weights from local cache only.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"  # Save path for checkpoints
    seed_everything(args.seed)  # Fix seeds for better reproducibility
    train_multitask(args)
    test_model(args)