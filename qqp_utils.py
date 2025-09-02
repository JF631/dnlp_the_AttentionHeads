# Utils script for qqp task, maybe merge into multitask_classifier on final merge?

import numpy as np
import torch

def pos_weight_from_labels(labels):
    """
    Return scalar pos_weight tensor for BCEWithLogitsLoss
    pos_weight = #neg / #pos
    """
    labels = np.asarray(labels, dtype=np.int64)
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    if pos == 0 or neg == 0:
        w = 1.0  # fix for rare and weird local gpu problem, can probably be removed when using cluster
    else:
        w = float(neg) / float(pos)
    return torch.tensor(w, dtype=torch.float32)


def smooth_targets(y, eps=0.05): # might need to adjust value later
    """
    Binary label smoothing for BCEWithLogits
    y in {0,1} -> y_smooth = y*(1-eps) + 0.5*eps
    """
    return y * (1.0 - eps) + 0.5 * eps

# Not really a qqp function, but rather bert. Located here to prevent merge conflicts. Maybe move later?
def freeze_bert_bottom_k(model, k_layers=6):
    """
    Freeze embeddings and the first k encoder layers of mini bert
    Pooler stays trainable
    """
    bert = model.bert

    # Freeze all embeddings + layer norm
    for mod in (bert.word_embedding, bert.pos_embedding, bert.tk_type_embedding, bert.embed_layer_norm):
        for p in mod.parameters():
            p.requires_grad = False

    # Encoder layers: index < k_layers frozen, rest trains
    for i, layer in enumerate(bert.bert_layers):
        trainable = (i >= k_layers)
        for p in layer.parameters():
            p.requires_grad = trainable

    # Keep pooler trainable
    for p in bert.pooler_dense.parameters():
        p.requires_grad = True
