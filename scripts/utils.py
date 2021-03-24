import numpy as np

def get_label_mappings():
    appear_mapping = {}
    for line in open('../data/appear_labels.txt'):
        key, val = line.strip().split('\t')
        appear_mapping[key] = int(val)
    
    grade_mapping = {}
    for line in open('../data/grade_labels.txt'):
        key, val = line.strip().split('\t')
        grade_mapping[key] = int(val)

    return appear_mapping, grade_mapping


def fold_split(annots, train_subjs, k):
    target_fold_size = len(annots)//k + 1
    folds = [[] for _ in range(k)]
    num_ims = np.zeros(k, dtype=int)
    for subj in train_subjs:
        num_im = len(annots[annots['SubjectID'] == subj])
        for i in range(k):
            if num_ims[i] < target_fold_size:
                folds[i].append(subj)
                num_ims[i] += num_im
                break
    print(num_ims)
    return folds, num_ims


    