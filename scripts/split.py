import pandas as pd
import numpy as np

TEST_PERCENT = 20

annots = pd.read_csv('../data/final_annotations.csv')
num_total = len(annots)
num_test = int(num_total * 20/100)
print(num_total)

subjs = list(pd.unique(annots['SubjectID']))
subj_idxs = np.arange(len(subjs))
np.random.shuffle(subj_idxs)
print(len(subjs))
test_subj = []
train_subj = []
num_test_curr = 0
num_test_subj = 0
for subj_idx in subj_idxs:
    if num_test_curr < num_test:
        annots_subj = annots[annots['SubjectID'] == subjs[subj_idx]]
        test_subj.append(subjs[subj_idx])
        num_test_curr += len(annots_subj)
        num_test_subj += 1
    else:
        train_subj.append(subjs[subj_idx])

test_subj = pd.DataFrame(test_subj, columns=["Test Subject ID"])
train_subj = pd.DataFrame(train_subj, columns=["Train Subject ID"])
test_subj.to_csv("../data/test_subjects.csv")
train_subj.to_csv("../data/train_subjects.csv")

print("Test subjects:", num_test_subj)
print("Test images:", num_test_curr)

    