import pandas as pd
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description='Count class distribution in k-fold splits')
parser.add_argument('--k', default='4', type=int, help="Number of folds")
parser.add_argument('--form', default='count', type=str, help="One of 'count', 'prob'")
args = parser.parse_args()

annots = pd.read_csv('../data/final_annotations.csv')
all_subj = open('../data/train_subjects.txt').read().splitlines()

fold_subjs = np.array_split(all_subj, args.k)
for fold_subj in fold_subjs:
    annots_fold = annots[annots['SubjectID'].isin(fold_subj)]
    print(len(annots_fold))
sys.exit()

appear_counts = annots['AppearLabel'].value_counts()
if args.form == 'prob': appear_counts /= len(annots)
grade_counts = annots['GradeLabel'].value_counts()
if args.form == 'prob': grade_counts /= len(annots)
joint_counts = annots.groupby(['AppearLabel', 'GradeLabel']).size()
if args.form == 'prob': joint_counts /= len(annots)

print("Appearance class:")
print(appear_counts)
print("\nGrade class:")
print(grade_counts)
print("\nJoint class:")
print(joint_counts)
