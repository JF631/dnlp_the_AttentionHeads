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
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW

import nlpaug.augmenter.word as naw
import nltk
import os

from transformers import MarianMTModel, MarianTokenizer
import json

# NLTK setup
NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.insert(0, NLTK_DIR)
nltk.download("wordnet", download_dir=NLTK_DIR)
nltk.download("omw-1.4", download_dir=NLTK_DIR)
nltk.download("averaged_perceptron_tagger_eng", download_dir=NLTK_DIR)


# Backtranslation models (lazy-loaded later)
EN_TO_FR = "Helsinki-NLP/opus-mt-en-fr"
FR_TO_EN = "Helsinki-NLP/opus-mt-fr-en"

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

        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        # General dropout layer using hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Sentiment classification layer
        # This layer will be used for the SST dataset.
        self.sentiment_classifier = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        # Attention pooling for sentiment
        self.sentiment_attention = nn.Linear(config.hidden_size, 1)

        self.sts_regressor = nn.Linear(self.bert.config.hidden_size * 2, 1)

        # Input is 2 * 768 (two sentance embeddings), output is 1 since it is single 0/1 (yes/no)
        self.paraphrase_classifier = nn.Linear(2 * BERT_HIDDEN_SIZE, 1)

    def forward(self, input_ids, attention_mask, return_all_tokens=False):
        """
        Takes a batch of sentences and produces embeddings for them.
        If return_all_tokens=True, returns all token embeddings (batch_size, seq_len, hidden_size).
        Else, returns pooler_output (batch_size, hidden_size).
        """

        # The final BERT embedding is the hidden state of [CLS] token (the first token).
        # See BertModel.forward() for more details.
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        bert_output = self.bert(input_ids, attention_mask)
        if return_all_tokens:
            return bert_output['last_hidden_state']
        return bert_output['pooler_output']

    def predict_sentiment(self, input_ids, attention_mask, pooling="mean"):
        """
        Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        Dataset: SST

        Different Pooling methods: 'cls', 'mean', 'max', 'attention'
        """

        if pooling == "cls":
            cls_embedding = self.forward(input_ids, attention_mask)
            pooled_embedding = self.dropout(cls_embedding)
        elif pooling == "mean":
            token_embeddings = self.forward(input_ids, attention_mask, return_all_tokens=True)  
            masked_token_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  
            sum_embeddings = masked_token_embeddings.sum(dim=1)  
            lengths = attention_mask.sum(dim=1).unsqueeze(-1)  
            mean_embedding = sum_embeddings / lengths.clamp(min=1e-9)  
            pooled_embedding = self.dropout(mean_embedding)
        elif pooling == "max":
            token_embeddings = self.forward(input_ids, attention_mask, return_all_tokens=True)  
            masked_token_embeddings = token_embeddings.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))  
            max_embedding, _ = masked_token_embeddings.max(dim=1)  
            pooled_embedding = self.dropout(max_embedding)
        elif pooling == "attention":
            token_embeddings = self.forward(input_ids, attention_mask, return_all_tokens=True)  
            attn_scores = self.sentiment_attention(token_embeddings) 
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=1) 
            pooled_embedding = (token_embeddings * attn_weights).sum(dim=1) 
            pooled_embedding = self.dropout(pooled_embedding)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        logits = self.sentiment_classifier(self.dropout(pooled_embedding))  
        return logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: Quora
        """
        # Embeddings for each sentences
        emb1 = self.forward(input_ids_1, attention_mask_1)
        emb2 = self.forward(input_ids_2, attention_mask_2)

        # Combine embeddings
        combined_emb = torch.cat((emb1, emb2), dim=1)

        # Apply dropout
        dropped_emb = self.dropout(combined_emb)

        # Make prediction
        logits = self.paraphrase_classifier(dropped_emb)

        return logits.squeeze(-1)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Since the similarity label is a number in the interval [0,5], your output should be normalized to the interval [0,5];
        it will be handled as a logit by the appropriate loss function.
        Dataset: STS
        """
        cls_1 = self.forward(input_ids_1, attention_mask_1)  # ?
        cls_2 = self.forward(input_ids_2, attention_mask_2)  # ?
        combined = torch.cat([cls_1, cls_2], dim=1)  # ?
        combined = self.dropout(combined)  # ?
        similarity = self.sts_regressor(combined)  # ?
        similarity = torch.sigmoid(similarity) * 5  # normalize to [0, 5] ?
        return similarity.view(-1)  # ?


def predict_paraphrase_types(
            self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs logits for detecting the paraphrase types.
        There are 7 different types of paraphrases.
        Thus, your output should contain 7 unnormalized logits for each sentence. It will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: ETPC
        """
        ### TODO
        raise NotImplementedError


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


# -------------------------
# Data Augmentation (for SST)
# -------------------------

def synonym_augment(data, aug_p=0.5):
    aug = naw.SynonymAug(aug_src="wordnet", aug_p=aug_p)
    augmented = []
    for sent, label, sent_id in data:
        aug_sent = aug.augment(sent)
        if isinstance(aug_sent, list):
            aug_sent = " ".join(aug_sent)
        augmented.append((aug_sent, label, sent_id + "_aug"))
    return augmented


def back_translate(sentence, en_to_fr_model, en_to_fr_tokenizer, fr_to_en_model, fr_to_en_tokenizer, device, max_length=128):

    # EN -> FR
    inputs = en_to_fr_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        translated = en_to_fr_model.generate(**inputs)
    fr_texts = en_to_fr_tokenizer.batch_decode(translated, skip_special_tokens=True)

    # FR -> EN
    inputs = fr_to_en_tokenizer(fr_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        back_translated = fr_to_en_model.generate(**inputs)
    en_texts = fr_to_en_tokenizer.batch_decode(back_translated, skip_special_tokens=True)

    return en_texts

# -------------------------

# TODO Currently only trains on SST dataset!
def train_multitask(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    # Create the data and its corresponding datasets and dataloader:
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train"
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
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
        
        # sst_train_data += synonym_augment(sst_train_data)
        # print(f"Total SST training data size after synonym augmentation: {len(sst_train_data)}")

        cache_file = "sst_backtranslated.json"

        if os.path.exists(cache_file):
            print(f"Loading cached back-translated data from {cache_file}")
            with open(cache_file, "r") as f:
                augmented_data = json.load(f)
        else:
            print("Generating back-translated SST data...")
            en_to_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
            fr_to_en_model_name = "Helsinki-NLP/opus-mt-fr-en"

            en_to_fr_tokenizer = MarianTokenizer.from_pretrained(en_to_fr_model_name)
            en_to_fr_model = MarianMTModel.from_pretrained(en_to_fr_model_name).to(device)
            fr_to_en_tokenizer = MarianTokenizer.from_pretrained(fr_to_en_model_name)
            fr_to_en_model = MarianMTModel.from_pretrained(fr_to_en_model_name).to(device)

            augmented_data = []
            batch_size_bt = 16
            num_batches = (len(sst_train_data) + batch_size_bt - 1) // batch_size_bt
            print(f"Back-translating {len(sst_train_data)} sentences in {num_batches} batches of size {batch_size_bt}...")

            for i in tqdm(range(0, len(sst_train_data), batch_size_bt), total=num_batches):
                batch = sst_train_data[i:i+batch_size_bt]
                batch_texts = [sent for sent, _, _ in batch]
                back_texts = back_translate(batch_texts,
                                            en_to_fr_model, en_to_fr_tokenizer,
                                            fr_to_en_model, fr_to_en_tokenizer,
                                            device)
                for (orig, label, sent_id), bt_sent in zip(batch, back_texts):
                    augmented_data.append((bt_sent, label, sent_id + "_bt"))
            with open(cache_file, "w") as f:
                json.dump(augmented_data, f)
            print(f"Cached {len(augmented_data)} augmented examples to {cache_file}")

        print(f"Generated {len(augmented_data)} back-translated examples for SST training data.")
        # Combine with original training data
        sst_train_data = sst_train_data + augmented_data
        print(f"Total SST training data size after back-translation: {len(sst_train_data)}")


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

    ### TODO
    #   Load data for the other datasets
    # If you are doing the paraphrase type detection with the minBERT model as well, make sure
    # to transform the the data labels into binaries (as required in the bart_detection.py script)
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
                loss = F.mse_loss(preds, labels.float())  # Regression loss for STS similarity in range [0, 5]
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "qqp" or args.task == "multitask":
            # Trains the model on the qqp dataset
            for batch in tqdm(quora_train_dataloader, desc=f"train-{epoch + 1:02}", disable=TQDM_DISABLE):
                # Move batch to device
                b_ids1, b_mask1, \
                    b_ids2, b_mask2, \
                    b_labels = (
                    batch['token_ids_1'].to(device), batch['attention_mask_1'].to(device),
                    batch['token_ids_2'].to(device), batch['attention_mask_2'].to(device),
                    batch['labels'].to(device)
                )

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)

                loss = F.binary_cross_entropy_with_logits(logits, b_labels.float().view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "etpc" or args.task == "multitask":
            # Trains the model on the etpc dataset
            ### TODO
            raise NotImplementedError

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

        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multitask": (0, 0),  # TODO
        }[args.task]

        print(
            f"Epoch {epoch + 1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)


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

    # TODO
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
