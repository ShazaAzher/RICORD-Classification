import numpy as np
from sklearn.metrics import average_precision_score

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
    
