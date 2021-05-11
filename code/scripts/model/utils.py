import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd
import os
import torch
from kornia.losses import focal_loss as focal

def get_label_mappings():
    mappings = {}
    mappings['appear'] = {}
    for line in open('../data/appear_labels.txt'):
        key, val = line.strip().split('\t')
        mappings['appear'][key] = int(val)
    
    mappings['grade'] = {}
    for line in open('../data/grade_labels.txt'):
        key, val = line.strip().split('\t')
        mappings['grade'][key] = int(val)

    return mappings


def fold_split(annots, train_subjs, k, asgt_mode='random'):
    target_fold_size = len(annots)//k + 1
    folds = [[] for _ in range(k)]
    num_ims = np.zeros(k, dtype=int)
    for subj in train_subjs:
        num_im = len(annots[annots['SubjectID'] == subj])
        fold_idxs = np.arange(k)
        if asgt_mode == 'random': np.random.shuffle(fold_idxs)
        
        for idx in fold_idxs:
            if num_ims[idx] < target_fold_size:
                folds[idx].append(subj)
                num_ims[idx] += num_im
                break

    return folds, num_ims


def multiclass_AP(targets, probs):
    return [average_precision_score((targets==c), probs[:,c]) for c in range(probs.shape[1])]


def get_best_val_loss(finetune_path, fold):
    history = pd.read_csv(os.path.join(finetune_path, 'fold_' + str(fold), 'history.csv'))
    return history['val_total_loss'].min()


def CE_loss_distr(outputs, targets, ignore_index=None):
    if ignore_index:
      keep_rows = (targets[:, 0] != ignore_index)
      loss = -torch.mean(targets[keep_rows] * torch.log(torch.nn.functional.softmax(outputs[keep_rows], dim=1)), dim=1)
    else:
      loss = -torch.mean(targets * torch.log(torch.nn.functional.softmax(outputs, dim=1)), dim=1)
    return torch.mean(loss)


def focal_loss(outputs, targets, focal_params, device, label_type):
  if label_type=='grade':
    mask = (targets != -1)
    outputs_masked = outputs[mask, :]
    targets_masked = targets[mask]
    if len(targets_masked):
      loss = focal(outputs_masked, targets_masked, \
        alpha=focal_params['alpha'], gamma=focal_params['gamma'], \
        reduction='mean')
    else:
      loss = torch.tensor([0], dtype=torch.float32, requires_grad=True)
      loss = loss.to(device)
  elif label_type=='appear':
    loss = focal(outputs, targets, \
        alpha=focal_params['alpha'], gamma=focal_params['gamma'], \
        reduction='mean')
  return loss
