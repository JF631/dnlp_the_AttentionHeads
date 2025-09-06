import argparse
import math
import random

import numpy as np
import pandas as pd
import ast
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from sklearn.metrics import matthews_corrcoef
# from optimizer import AdamW
from torch.optim import AdamW

# Only for evaluation purposes
from sklearn.metrics import f1_score
from torch.utils.data import Sampler

import json, time, os

TQDM_DISABLE = False

def save_run(out_prefix,
             probs_dev, labels_dev,
             macro_f1_hist, micro_f1_hist, train_loss_hist, dev_loss_hist,
             summary, meta):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    np.savez_compressed(
        out_prefix + ".npz",
        probs_dev=probs_dev,
        labels_dev=labels_dev,
        macro_f1_hist=np.asarray(macro_f1_hist),
        micro_f1_hist=np.asarray(micro_f1_hist),
        train_loss_hist=np.asarray(train_loss_hist),
        dev_loss_hist=np.asarray(dev_loss_hist)
    )
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,   # {"acc":..., "mcc":..., "macro_f1":...}
        "meta": meta
    }
    with open(out_prefix + ".json", "w") as f:
        json.dump(payload, f, indent=2)



class MultiLabelBalancedBatchSampler(Sampler):
    """
    Oversamples rare labels so that each batch has a balanced mix of frequent and rare labels.
    Useful for multi-label classification with long-tail distributions.
    """

    def __init__(self, label_matrix, batch_size=32, labels_per_batch=4, samples_per_label=8,
                 drop_last=False, seed=11711, rare_boost=True):
        """
        Args:
            label_matrix: [N, C] numpy array or torch tensor with {0,1} multi-hot labels
            batch_size: total samples per batch
            labels_per_batch: how many distinct labels to target per batch
            samples_per_label: target samples per chosen label
            rare_boost: if True, oversample rare labels with higher probability
        """
        if torch.is_tensor(label_matrix):
            label_matrix = label_matrix.cpu().numpy()
        self.N, self.C = label_matrix.shape
        self.batch_size = batch_size
        self.labels_per_batch = labels_per_batch
        self.samples_per_label = samples_per_label
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        self.idx_per_label = []
        for c in range(self.C):
            idx = np.where(label_matrix[:, c] == 1)[0].tolist()
            self.idx_per_label.append(idx)

        # only labels that actually appear
        self.active_labels = [c for c in range(self.C) if len(self.idx_per_label[c]) > 0]

        # label frequencies
        freqs = np.array([len(self.idx_per_label[c]) for c in self.active_labels], dtype=float)
        if rare_boost:
            self.label_probs = (1.0 / (freqs + 1e-6))
            self.label_probs /= self.label_probs.sum()
        else:
            self.label_probs = np.ones_like(freqs) / len(freqs)

        # estimated number of batches per epoch
        self._num_batches = self.N // self.batch_size if drop_last else math.ceil(self.N / self.batch_size)

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            chosen = self.rng.choices(
                self.active_labels,
                weights=self.label_probs,
                k=min(self.labels_per_batch, len(self.active_labels))
            )

            pool = []
            for c in chosen:
                idxs = self.idx_per_label[c]
                if len(idxs) == 0:
                    continue
                k = min(self.samples_per_label, len(idxs))
                if len(idxs) < k:
                    pool += self.rng.choices(idxs, k=k)
                else:
                    pool += self.rng.sample(idxs, k=k)

            # de-duplicate
            seen = set()
            batch = []
            for i in pool:
                if i not in seen:
                    seen.add(i)
                    batch.append(i)
                if len(batch) >= self.batch_size:
                    break

            # top up randomly if short
            if len(batch) < self.batch_size:
                remainder = [i for i in range(self.N) if i not in seen]
                if len(remainder) > 0:
                    fill = self.rng.sample(remainder, 
                                           k=min(self.batch_size - len(batch), len(remainder)))
                    batch += fill

            yield batch

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos, self.gamma_neg, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps
    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps)) * ((1 - xs_pos) ** self.gamma_pos)
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps)) * (xs_pos ** self.gamma_neg)
        loss = -(loss_pos + loss_neg)
        return loss.mean()

class SupConLoss(nn.Module):
    def __init__(self, temp=0.05, use_jaccard=True, hard_neg_margin=None, eps=1e-12):
        """
        temp: temperature for scaling similarities
        use_jaccard: if True, weight positives by Jaccard(label_i, label_j)
        hard_neg_margin: if set (e.g. 0.2), up-weights negatives with sim > margin
        """
        super().__init__()
        self.temp = temp
        self.use_jaccard = use_jaccard
        self.hard_neg_margin = hard_neg_margin
        self.eps = eps

    def forward(self, features, labels):
        device = features.device
        B = labels.size(0)

        f = nn.functional.normalize(features, dim=1)
        # cosine similarity / temperature
        logits = (f @ f.T) / max(self.temp, self.eps)

        eye = torch.eye(B, device=device, dtype=torch.bool)
        logits = logits.masked_fill(eye, -1e9)

        overlap = (labels @ labels.T).float()
        size = labels.sum(1, keepdim=True).float()
        union = size + size.T - overlap

        if self.use_jaccard:
            w_pos = torch.where(union > 0, overlap / (union + self.eps), torch.zeros_like(union))
        else:
            w_pos = (overlap > 0).float()

        # remove self
        w_pos = w_pos.masked_fill(eye, 0.0)

        logits = logits - logits.max(dim=1, keepdim=True).values
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        denom = w_pos.sum(1)
        valid = denom > 0

        loss_i = torch.zeros(B, device=device)
        loss_i[valid] = -(w_pos[valid] * log_prob[valid]).sum(1) / (denom[valid] + self.eps)

        if self.hard_neg_margin is not None:
            neg_mask = (w_pos == 0).float() * (~eye).float()
            hard_neg = (logits > self.hard_neg_margin).float() * neg_mask
            hn_denom = hard_neg.sum(1).clamp_min(1.0)
            hn_term = (hard_neg * torch.exp(log_prob)).sum(1) / hn_denom
            loss_i = loss_i + 0.1 * hn_term

        return loss_i.mean()



class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26, projection_dim=128, use_optim=True):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=True)
        if use_optim:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.bart.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_labels)
            )
        else:
            self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)

        self.projection = nn.Sequential(
            nn.Linear(self.bart.config.hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, input_ids, attention_mask):
        out = self.bart(input_ids, attention_mask)
        last_hidden_state = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()  # shape: [batch, seq_len, hidden]
        summed = torch.sum(last_hidden_state * mask, dim=1)
        summed_mask = mask.sum(dim=1)
        mean_pooled = summed / torch.clamp(summed_mask, min=1e-9)

        projected = self.projection(mean_pooled)
        return mean_pooled, projected
 
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # cls_output = last_hidden_state[:, 0, :]

        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        summed_mask = mask.sum(dim=1)
        mean_pooled = summed / torch.clamp(summed_mask, min=1e-9)

        # Add an additional fully connected layer to obtain the logits
        # logits = self.classifier(cls_output)

        # Return the probabilities
        # probabilities = self.sigmoid(logits)
        # return logits

        # mean_pooled, _ = self.encode(input_ids, attention_mask)
        logits = self.classifier(mean_pooled)
        return logits


def transform_data(dataset: pd.DataFrame, max_length=512, shuffle=True, return_dataset=False):
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [2, 5, 6, 0, 0, 0, 0]. This means that
    the sentence pair is of type 2, 5, and 6. Turn this into a binary form, where the
    label becomes [0, 1, 0, 0, 1, 1, 0]. Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """
    # raise NotImplementedError
    dataset = dataset.copy()
    col_keys = list(dataset.columns)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    sentences1 = dataset['sentence1'].astype('str').tolist()
    sentences2 = dataset['sentence2'].astype('str').tolist()

    inputs = tokenizer(sentences1, sentences2,
                       padding=True,
                       truncation=True,
                       max_length = max_length,
                       return_tensors='pt') #pt = pytorch

    # inputs1 = tokenizer(sentences1, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    # inputs2 = tokenizer(sentences2, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    # input_ids2, attention_mask2 = inputs2['input_ids'], inputs2['attention_mask']
    unused_ids = [12, 19, 20, 23, 27]
    valid_labels = sorted(set(range(1, 32)) - set(unused_ids))
    label_map = {label: idx for idx, label in enumerate(valid_labels)} 
    if "paraphrase_type_ids" in col_keys:
        dataset['paraphrase_type_ids'] = dataset['paraphrase_type_ids'].apply(ast.literal_eval)
        binarized_labels = np.zeros((len(sentences1), len(valid_labels)), dtype='i4')
        for i, label_list in enumerate(dataset['paraphrase_type_ids']):
            for label in label_list:
                if label in label_map:
                    binarized_labels[i, label_map[label]] = 1
        binarized_labels = torch.tensor(binarized_labels, dtype=torch.float)
        tds = TensorDataset(input_ids, attention_mask, binarized_labels)
    else:
        tds = TensorDataset(input_ids, attention_mask)
    
    if return_dataset:
        return tds, (binarized_labels if "paraphrase_type_ids" in col_keys else None)

    return DataLoader(tds, batch_size=32, shuffle=shuffle, num_workers=0)

    # return DataLoader(dataset, batch_size=32, shuffle=shuffle, num_workers=0)

def evaluate_on_dev(model, device, loss_fn, current_batch):
    dev_losses = []
    if len(current_batch) != 3:
        raise RuntimeError(f"Expected batch size 3 (ids, mask, labels) but got {len(current_batch)}") 
    dev_ids, dev_mask, dev_labels = [b.to(device) for b in current_batch]
    probs = model(dev_ids, dev_mask)
    loss = loss_fn(probs, dev_labels.float())
    dev_losses.append(loss.item())
    return dev_losses

def collect_probs_labels(model, data_loader, device):
    """
    Runs the model on a loader of (input_ids, attention_mask, labels) batches and
    returns:
        probs  : np.ndarray of shape [N, C] (sigmoid applied)
        labels : np.ndarray of shape [N, C]
    Skips batches that don't contain labels (len(batch) != 3).
    """
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) != 3:
                continue
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if not all_probs:  # no labeled batches
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    probs_np = torch.vstack(all_probs).numpy()
    labels_np = torch.vstack(all_labels).numpy()
    return probs_np, labels_np

def train_model(model, train_data, dev_data, device, loss_fn = None, patience=3, min_delta=0.0, loss_weight=0.01):
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.

    Return the trained model.
    """
    with open(args.out, 'a+') as outfile:
        outfile.write(f"Training model with lossweight={loss_weight}\n")
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    if args.use_optim:
        loss_fn = AsymmetricLoss(gamma_pos=0, gamma_neg=4)
        contr_loss_fn = SupConLoss(temp=0.05, use_jaccard=True, hard_neg_margin=0.2)
        lamda_contr_loss = loss_weight
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        print("using bce loss")

    n_epochs = 12
    epoch_losses = np.empty((n_epochs,))
    dev_epoch_losses = np.empty((n_epochs,))

    macro_f1_hist, micro_f1_hist = [], []

    for epoch in tqdm(range(n_epochs), desc="Running training loop"):
        model.train()
        losses = []
        for batch in tqdm(train_data, desc=f"Epoch {epoch + 1}", leave=False):
            if len(batch) == 3:
                in_ids, attention_mask, labels = [b.to(device) for b in batch]
            else:
                continue
            #forward classification pass
            probs = model(in_ids, attention_mask)
            #focal loss
            cls_loss = loss_fn(probs, labels.float())

            #contrastive loss
            if args.use_optim:
                _, projection  = model.encode(in_ids, attention_mask)
                contr_loss = contr_loss_fn(projection, labels)

                loss = cls_loss + lamda_contr_loss * contr_loss
            else:
                loss = cls_loss

            #backward pass 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_losses[epoch] = np.mean(losses)
        acc_train, mcc_train = evaluate_model(model, train_data, device) 
        print (f"epoch {epoch}: training loss: {epoch_losses[epoch]}, acc: {acc_train}, mcc: {mcc_train}\n")

        losses.clear()
        model.eval()
        dev_losses = []
        with torch.no_grad():
            for dev_batch in dev_data:
                if len(dev_batch) != 3:
                    continue
                din_ids, dattention_mask, dlabels = [b.to(device) for b in dev_batch]
                dprobs = model(din_ids, dattention_mask)
                dloss = loss_fn(dprobs, dlabels.float())
                dev_losses.append(dloss.item())
        cur_dev = float(np.mean(dev_losses)) if dev_losses else np.nan
        dev_epoch_losses[epoch] = cur_dev
        
        probs_dev_epoch, labels_dev_epoch = collect_probs_labels(model, dev_data, device)
        macro05 = f1_score(labels_dev_epoch, (probs_dev_epoch > 0.5).astype(int),
                           average='macro', zero_division=0)
        micro05 = f1_score(labels_dev_epoch, (probs_dev_epoch > 0.5).astype(int),
                           average='micro', zero_division=0)
        macro_f1_hist.append(float(macro05))
        micro_f1_hist.append(float(micro05))

        acc_dev, mcc_dev = evaluate_model(model, dev_data, device)
        with open(args.out, 'a+') as outfile:
            outfile.write(
                f"epoch {epoch}: dev loss: {dev_epoch_losses[epoch]:.4f}, "
                f"acc: {acc_dev:.4f}, mcc: {mcc_dev:.4f}\n"
            )
    return dev_epoch_losses, epoch_losses, macro_f1_hist, micro_f1_hist



def test_model(model, test_data, test_ids, device, thresholds=None):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    threshold = 0.5
    model = model.to(device)
    model.eval()
    model_predictions = []
    with torch.no_grad():
        for batch in test_data:
            if len(batch) == 3:
                token_ids, attention_mask, _ = [val.to(device) for val in batch]
            elif len(batch) == 2:
                token_ids, attention_mask = [val.to(device) for val in batch]
            else:
                raise RuntimeError(f"Expected 2 or 3 values in batch, but got {len(batch)}")
            raw_out = model(token_ids, attention_mask)
            probs = torch.sigmoid(raw_out)
            if thresholds is None:
                filtered_out = (probs > threshold).int().cpu().tolist()
            else:
                filtered_out = []
                for row in probs.cpu().numpy():
                    filtered_out.append([(1 if row[i] > thresholds[i] else 0) for i in range(len(thresholds))])
            model_predictions.extend(filtered_out)

    rtrn = pd.DataFrame({
        'id':test_ids,
        'Predicted_Paraphrase_Types':model_predictions
    }) 
    return rtrn

def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point.
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            predicted_labels = (probs > 0.5).int()
            # print(predicted_labels)

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    # Compute the accuracy for each label
    accuracies = []
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        #compute Matthwes Correlation Coefficient for each paraphrase type
        matth_coef = matthews_corrcoef(true_labels_np[:,label_idx], predicted_labels_np[:,label_idx])
        matthews_coefficients.append(matth_coef)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    matthews_coefficient = np.mean(matthews_coefficients)
    model.train()
    return accuracy, matthews_coefficient


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
    parser.add_argument("--out", type=str, default="out.txt")
    parser.add_argument("--weight", type=float, default=0.01)
    parser.add_argument("--use_optim", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    if device == 'cpu':
        raise RuntimeWarning("Training runs on CPU!")
    model = BartWithClassifier(use_optim=args.use_optim).to(device)
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep=",")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep=",")
    train_dataset = train_dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_idx = int(0.8 * len(train_dataset))
    train_split = train_dataset[:split_idx]
    validation_split = train_dataset[split_idx:]
    
    train_ds, train_labels = transform_data(train_split, return_dataset=True)
    dev_data = transform_data(validation_split, shuffle=False)
    test_data = transform_data(test_dataset, shuffle=False)
    
    if args.use_optim:
        train_sampler = MultiLabelBalancedBatchSampler(
            label_matrix=train_labels,  # [N, 26]
            batch_size=32,
            labels_per_batch=6,
            samples_per_label=8,
            drop_last=False,
            seed=args.seed
        )

        train_data = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
    else:
        train_data = transform_data(train_split, shuffle=True)
    print(f"Loaded {len(train_dataset)} training samples.")


    dev_losses, train_losses,  macro_f1_hist, micro_f1_hist = train_model(
        model, train_data, dev_data, device, patience=3, loss_weight=args.weight)
    
    # Ouptut saving
    final_probs_dev, final_labels_dev = collect_probs_labels(model, dev_data, device)

    print(f"dev losses: {dev_losses}")
    print(f"losses: {train_losses}")

    print("Training finished, evaluating model performance...")
    accuracy, matthews_corr= evaluate_model(model, dev_data, device)
    print(f"ACC: {accuracy:.3f}")
    print(f"MCC: {matthews_corr:.3f}")
    print("done")
    with open(args.out, 'a+') as outfile:
        outfile.write(
            f"ACC: {accuracy:.3f}, "
            f"MCC: {matthews_corr:.3f}, "
        )
    
    summary = {"acc": float(accuracy), "mcc": float(matthews_corr)}
    meta = {
        "use_optim": args.use_optim,
        "seed": args.seed,
        "weight": args.weight,
        "projection_dim": 128,
        "temp": 0.05,
        "sampler": {"labels_per_batch": 6, "samples_per_label": 6, "batch_size": 32}
    }

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_prefix = f"runs/etpc_{'optim' if args.use_optim else 'baseline'}_{run_id}"

    save_run(out_prefix,
            probs_dev=final_probs_dev,
            labels_dev=final_labels_dev,
            macro_f1_hist=macro_f1_hist,
            micro_f1_hist=micro_f1_hist,
            train_loss_hist=train_losses,
            dev_loss_hist=dev_losses,
            summary=summary,
            meta=meta)

    print(f"[saved] {out_prefix}.npz and {out_prefix}.json")
    
    # Test output for subission
    filename = "predictions/bart/etpc-paraphrase-detection-test-output.csv"
    print(f"Testing the model and saving to {filename}")
    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device, thresholds=None)
    test_results.to_csv(
        filename, index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
