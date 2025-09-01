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
