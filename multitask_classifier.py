import argparse
import os
from pprint import pformat
import random
import re
import sys
import time
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
from datasets import (
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


# fix the random seed
def seed_everything(seed=11711):
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
    This module should use BERT for these tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    (- Paraphrase type detection (predict_paraphrase_types))
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        # Choose which model to use
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

        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        # General dropout layer using hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Heads
        self.sentiment_classifier = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        self.sts_regressor = nn.Linear(self.bert.config.hidden_size * 2, 1)

        # Input is 2 * 768 (two sentence embeddings)
        self.paraphrase_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)
        # ETPC paraphrase type detection head (7 binary labels, multi-label)
        self.paraphrase_types_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, 14)

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return bert_output['pooler_output']

    def predict_sentiment(self, input_ids, attention_mask):
        cls_embedding = self.forward(input_ids, attention_mask)
        logits_after_dropout = self.dropout(cls_embedding)
        logits = self.sentiment_classifier(logits_after_dropout)
        return logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        emb1 = self.forward(input_ids_1, attention_mask_1)
        emb2 = self.forward(input_ids_2, attention_mask_2)
        combined_emb = torch.cat((emb1, emb2), dim=1)
        dropped_emb = self.dropout(combined_emb)
        logits = self.paraphrase_classifier(dropped_emb)
        return logits.squeeze(-1)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        cls_1 = self.forward(input_ids_1, attention_mask_1)
        cls_2 = self.forward(input_ids_2, attention_mask_2)
        combined = torch.cat([cls_1, cls_2], dim=1)
        combined = self.dropout(combined)
        similarity = self.sts_regressor(combined)
        similarity = torch.sigmoid(similarity) * 5  # normalize to [0, 5]
        return similarity.view(-1)

    def predict_paraphrase_types(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        emb1 = self.forward(input_ids_1, attention_mask_1)
        emb2 = self.forward(input_ids_2, attention_mask_2)
        combined_emb = torch.cat((emb1, emb2), dim=1)
        dropped_emb = self.dropout(combined_emb)
        logits = self.paraphrase_types_classifier(dropped_emb)
        return logits


def save_model(model, optimizer, args, config, filepath):
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
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader:
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

    # SST dataset
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

    # STS dataset
    if args.task in ("sts", "multitask"):
        sts_train_data = SentencePairDataset(sts_train_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args)

        sts_train_dataloader = DataLoader(
            sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn
        )

    # QQP dataset
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

    # ETPC dataset  <<< ADDED
    if args.task == "etpc" or args.task == "multitask":
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

    # Init model
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

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    best_dev_acc = float("-inf")
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.task == "sst" or args.task == "multitask":
            # Train the model on the sst dataset.

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

                # Handling different Optimizer
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
            # Trains the model on the sts dataset
            for batch in tqdm(sts_train_dataloader, desc=f"train-{epoch + 1:02}-sts", disable=TQDM_DISABLE):
                input_ids_1, attention_mask_1 = batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device)
                input_ids_2, attention_mask_2 = batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad()
                preds = model.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = F.mse_loss(preds, labels.float())

                print(f"Using {args.optimizer} optimizer for {args.task}.")
                # Handling different Optimizer
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
            # Trains the model on the qqp dataset
            for batch in tqdm(quora_train_dataloader, desc=f"train-{epoch + 1:02}", disable=TQDM_DISABLE):
                # Move batch to a device
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                    batch['token_ids_1'].to(device), batch['attention_mask_1'].to(device),
                    batch['token_ids_2'].to(device), batch['attention_mask_2'].to(device),
                    batch['labels'].to(device)
                )

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.float().view(-1))

                # Handling different optimizer
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

        if args.task == "etpc" or args.task == "multitask":
            # Train on ETPC (7-label multi-label classification)
            for batch in tqdm(etpc_train_dataloader, desc=f"train-{epoch + 1:02}-etpc", disable=TQDM_DISABLE):
                b_ids1 = batch["token_ids_1"].to(device)
                b_mask1 = batch["attention_mask_1"].to(device)
                b_ids2 = batch["token_ids_2"].to(device)
                b_mask2 = batch["attention_mask_2"].to(device)
                b_labels = batch["labels"].to(device).float()  # shape [B,10]

                optimizer.zero_grad()
                logits = model.predict_paraphrase_types(b_ids1, b_mask1, b_ids2, b_mask2)  # [B,7]
                loss = F.binary_cross_entropy_with_logits(logits, b_labels)

                # Handling different optimizer
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

        train_loss = train_loss / num_batches

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

        # Aggregate metrics selection
        if args.task == "multitask":
            dev_parts = []
            train_parts = []

            if sst_dev_dataloader is not None:
                dev_parts.append(sst_dev_acc)  # accuracy in [0,1]
                train_parts.append(sst_train_acc)

            if quora_dev_dataloader is not None:
                dev_parts.append(quora_dev_acc)  # accuracy in [0,1]
                train_parts.append(quora_train_acc)

            if etpc_dev_dataloader is not None:
                dev_parts.append(etpc_dev_acc)  # accuracy/F1 per your evaluator
                train_parts.append(etpc_train_acc)

            if sts_dev_dataloader is not None:
                # Map Spearman [-1,1] to [0,1] for comparability
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

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        # Error Handling for MultiTask Optimizers (typo fixed)
        if args.optimizer in ('pcgrad', 'gradvac') and args.task != "multitask":
            print(f"The Optimizer {args.optimizer} is only for multitasks.")
            exit(1)


def test_model(args):
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
    parser = argparse.ArgumentParser()

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
        default="sst",
    )

    # Model configuration
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )

    # Optimizer Arguments
    parser.add_argument(
        "--optimizer",
        type=str,
        help="The optimizer to use for training",
        choices=["pcgrad", "gradvac"],
        default="",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use",
        choices=("bert", "convBert", "simBert"),  # Either standard Bert, convolutional Bert or siamese Bert
        default="bert"
    )

    parser.add_argument("--use_gpu", action="store_true")

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
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-detection-test-student.csv")

    parser.add_argument(
        "--etpc_test", type=str, default="data/etpc-paraphrase-detection-test-student.csv"
    )

    # Output paths
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-test-output.csv"
        ),
    )

    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-test-output.csv"
        ),
    )

    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-output.csv"
        ),
    )

    parser.add_argument(
        "--etpc_dev_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--etpc_test_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-test-output.csv"
        ),
    )

    # Hyperparameters
    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
