import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Count class distribution in dataset')
parser.add_argument('--set', default='all', type=str, help="One of 'train', 'test', 'all'")
parser.add_argument('--form', default='count', type=str, help="One of 'count', 'prob'")
args = parser.parse_args()

annots = pd.read_csv('../data/final_annotations.csv')

if args.set != 'all':
    subjs = open('../data/{}_subjects.txt'.format(args.set)).read().splitlines()
    annots = annots[annots['SubjectID'].isin(subjs)]

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
