import argparse
import random
import os 

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from optimizer import AdamW


TQDM_DISABLE = False

#for testing a new score, that discriminates against copying the phrase 
def compute_ibleu(hypotheses, references, sources, alpha=0.9):
    """
    hypotheses, references, sources: lists of strings of the same length
    Returns iBLEU on the 0..100 sacrebleu scale.
    """
    bleu = BLEU(effective_order=True)  # more stable on short texts
    bleu_ref = bleu.corpus_score(hypotheses, [references]).score
    bleu_src = bleu.corpus_score(hypotheses, [sources]).score
    ibleu = alpha * bleu_ref - (1 - alpha) * bleu_src
    return ibleu, bleu_ref, bleu_src

#to try k_drop a way of stabilazing training with noice, which is said to be effective at low IBLue scores 
def rdrop_kl_loss(logits1, logits2, labels, ignore_index=-100):
    """
    logits*: (B, T, V), labels: (B, T)
    Returns mean symmetric KL over non-ignored tokens.
    """
    logp1 = F.log_softmax(logits1, dim=-1)  # (B,T,V)
    logp2 = F.log_softmax(logits2, dim=-1)  # (B,T,V)

    # Stable KL in log-space
    kl12 = F.kl_div(logp1, logp2, log_target=True, reduction="none").sum(dim=-1)  # (B,T)
    kl21 = F.kl_div(logp2, logp1, log_target=True, reduction="none").sum(dim=-1)  # (B,T)
    kl = 0.5 * (kl12 + kl21)

    mask = (labels != ignore_index).float()  # (B,T)
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom

def transform_data(dataset, max_length=256,  shuffle = False):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase types.
    Return Data Loader.
    """
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    local_model_path = "/user/fabian.kathe/u17494/.cache/huggingface/hub/models--facebook--bart-large/snapshots/cb48c1365bd826bd521f650dc2e0940aee54720c"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    input_texts = []
    target_texts = []

    for _, row in dataset.iterrows():
        sentence1 = str(row["sentence1"])
        segment = str(row.get("sentence1_segment_location", ""))
        types = str(row.get("paraphrase_type_ids", "")) 

        input_text = f"{sentence1} </s> {segment} </s> {types}"
        input_texts.append(input_text)

        # Fallback to dummy target if sentence2 is not available
        if "sentence2" in row and pd.notna(row["sentence2"]):
            target_texts.append(str(row["sentence2"]))
        else:
            target_texts.append("DUMMY")

    inputs = tokenizer(
        input_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    targets = tokenizer(
        target_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss

    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=8, shuffle=shuffle)

def train_model(model, train_data, dev_df, device, tokenizer,
                num_epochs=5, base_lr=5e-5, warmup_ratio=0.1, rdrop_lambda=1):
    model.train()

    # --- checkpoint path ---
    ckpt_path = "checkpoints/best_bleu.pt"
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    # --- optimizer & scheduler ---
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=base_lr)
    steps_per_epoch = len(train_data)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"Using linear LR schedule with {warmup_steps}/{total_steps} warmup steps")

    # --- early-stopping settings ---
    min_delta = 0.1
    patience = 4
    best_bleu = float("-inf")
    epochs_no_improve = 0

    # (optional) seed a valid checkpoint so restore always works
    torch.save(model.state_dict(), ckpt_path)

    # sanity
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {num_trainable}")
    assert num_trainable > 0, "No trainable parameters!"

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0

        for batch in tqdm(train_data, desc="Training", disable=TQDM_DISABLE):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            #outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Skip batches with no supervised tokens (all -100)
            if (labels != -100).sum().item() == 0:
                # print("Skipping batch with no valid labels")
                continue

            # Two stochastic passes
            out1 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Average CE loss
            ce = 0.5 * (out1.loss + out2.loss)

            # Consistency loss (symmetric KL between token dists)
            kl = rdrop_kl_loss(out1.logits, out2.logits, labels) 

            # Total
            loss = ce + rdrop_lambda * kl

            total_loss += loss.item()


            #loss = outputs.loss # not needed with kdrop
            #total_loss += loss.item() # '' 

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_data)
        print(f"Average training loss: {avg_loss:.4f}")

        # ---- evaluate (returns iBLEU, BLEU_ref, BLEU_src) ----
        val_ibleu, val_bleu, val_bleu_src = evaluate_model(model, dev_df, device, tokenizer, alpha=0.9)
 
        print(f"Dev BLEU: {val_bleu:.2f} | BLEU(hyp,src): {val_bleu_src:.2f} | iBLEU@0.9: {val_ibleu:.2f}")

        # ---- early stopping on BLEU(hyp, ref) ----
        if val_bleu > best_bleu + min_delta:
            best_bleu = val_bleu
            torch.save(model.state_dict(), ckpt_path)
            epochs_no_improve = 0
            print(f"↑ New best BLEU ({best_bleu:.2f}). Saved {ckpt_path}.")
        else:
            epochs_no_improve += 1
            print(f"No BLEU improvement ({epochs_no_improve}/{patience}).")
            if epochs_no_improve >= patience:
                print("Early stopping — restoring best BLEU weights.")
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                return model

    print("Training done — restoring best BLEU weights.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model

def freeze_all_but_last_two_decoder_layers(model):
    """
    Freezes all parameters in the model except:
    - The last two decoder layers
    - The lm_head (output layer)
    """
    for name, param in model.named_parameters():
        if not (
            name.startswith("model.decoder.layers.10") or
            name.startswith("model.decoder.layers.11") or
            name.startswith("lm_head")
        ):
            param.requires_grad = False
    print("✅ Only top 2 decoder layers and lm_head are trainable.")

def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    ### TODO
    model.eval()
    predictions = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data, desc="Generating", disable=TQDM_DISABLE)):
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            decoded_outputs = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]

            decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            #for j in range(len(decoded_outputs)):
            #    index = i * input_ids.size(0) + j  # actual global index in test_ids
            #    print(f"\nDEBUG SAMPLE {index}")
            #    print("ID:            ", test_ids.iloc[index])
            #    print("Input sentence:", decoded_inputs[j])
            #   print("Generated:     ", decoded_outputs[j])

            predictions.extend(decoded_outputs)

    # Ensure test_ids is aligned
    return pd.DataFrame({
        "id": test_ids[:len(predictions)],
        "Generated_sentence2": predictions
    })


def evaluate_model(model, eval_df, device, tokenizer, alpha=0.9, max_length=50, num_beams=5):
    """
    eval_df must have columns 'sentence1' (source) and 'sentence2' (reference).
    Returns iBLEU (0..100).
    """
    model.eval()
    dataloader = transform_data(eval_df, shuffle=False)

    hyps = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = [t.to(device) for t in batch]
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
            hyps.extend([
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ])

    refs = eval_df["sentence2"].astype(str).tolist()
    srcs = eval_df["sentence1"].astype(str).tolist()

    ibleu, bleu_ref, bleu_src = compute_ibleu(hyps, refs, srcs, alpha=alpha)
    print(f"BLEU(hyp, ref): {bleu_ref:.2f} | BLEU(hyp, src): {bleu_src:.2f} | iBLEU@{alpha}: {ibleu:.2f}")

    model.train()
    return ibleu, bleu_ref, bleu_src

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    local_model_path = "/user/fabian.kathe/u17494/.cache/huggingface/hub/models--facebook--bart-large/snapshots/cb48c1365bd826bd521f650dc2e0940aee54720c"

    model = BartForConditionalGeneration.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    #model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)
    #tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)
    #freeze_all_but_last_two_decoder_layers(model)
    #train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    #dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")

    # You might do a split of the train data into train/validation set here
    # ...
    # Load and split training data into train/dev
    full_train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    full_train_dataset = full_train_dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_idx = int(0.9 * len(full_train_dataset))
    train_dataset = full_train_dataset[:split_idx]
    # for testing 
    #train_dataset = train_dataset.sample(n=4, random_state=42)
    dev_dataset = full_train_dataset[split_idx:]
    #dev_dataset = dev_dataset.sample(n=2, random_state=42)  # Or even n=2

    train_data = transform_data(train_dataset, shuffle = True)
    dev_data = transform_data(dev_dataset)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_dataset, device, tokenizer)

    print("Training finished.")

    final_ibleu, final_bleu, final_bleu_src = evaluate_model(model, dev_dataset, device, tokenizer, alpha=0.9)
    print(f"Final Dev BLEU: {final_bleu:.3f} | iBLEU: {final_ibleu:.3f} | BLEU(hyp,src): {final_bleu_src:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
