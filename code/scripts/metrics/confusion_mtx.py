import pandas as pd 
import numpy as np 
import argparse
import os
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import glob

parser = argparse.ArgumentParser(description='Computes confusion matrix')
parser.add_argument('model_folder', type=str, help='Folder containing model predictions')
parser.add_argument('--fallback_folder', type=str, help='Folder to use if best model doesn\'t exist (finetuning case)')
args = parser.parse_args()

appear_targets = []
appear_probs = []
grade_targets = []
grade_probs = []
for subfolder in os.listdir(args.model_folder):
  model_folder = args.model_folder if len(glob.glob(os.path.join(args.model_folder, subfolder, '*best*'))) \
    else args.fallback_folder
  appear_targets.append(np.load(os.path.join(model_folder, subfolder, 'appear_targets.npy')))
  appear_probs.append(np.load(os.path.join(model_folder, subfolder, 'appear_probs.npy')))
  grade_targets.append(np.load(os.path.join(model_folder, subfolder, 'grade_targets.npy')))
  grade_probs.append(np.load(os.path.join(model_folder, subfolder, 'grade_probs.npy')))

appear_targets = np.concatenate(appear_targets, axis=0)
appear_probs = np.concatenate(appear_probs, axis=0)
grade_targets = np.concatenate(grade_targets, axis=0)
grade_probs = np.concatenate(grade_probs, axis=0)

appear_preds = np.argmax(appear_probs, axis=1)
grade_preds = np.argmax(grade_probs, axis=1)

print("Appear confusion matrix:")
print(confusion_matrix(appear_targets, appear_preds))
print()
print("Grade confusion matrix:")
print(confusion_matrix(grade_targets, grade_preds))
print()
print("Appear accuracy:", top_k_accuracy_score(appear_targets, appear_probs, k=1))
print("Grade accuracy:", top_k_accuracy_score(grade_targets, grade_probs, k=1))
print()
print("Appear top-2 accuracy:", top_k_accuracy_score(appear_targets, appear_probs, k=2))
print("Grade top-2 accuracy:", top_k_accuracy_score(grade_targets, grade_probs, k=2))
  