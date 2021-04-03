import pandas as pd 
import numpy as np 
import argparse
import os

parser = argparse.ArgumentParser(description='Print average metrics over folds')
parser.add_argument('model_folder', type=str, help='Folder containing subfolders for each fold')
args = parser.parse_args()

all_best_metrics = []
for subfolder in os.listdir(args.model_folder):
  metrics = pd.read_csv(os.path.join(args.model_folder, subfolder, 'history.csv'), index_col=0)
  best_metrics = metrics.iloc[metrics['val_total_loss'].idxmin()]
  all_best_metrics.append(best_metrics)

all_best_metrics = pd.DataFrame(all_best_metrics)
avg_best_metrics = all_best_metrics.mean(axis=0)
print('Train appear mAP:', avg_best_metrics['train_appear_mAP'])
print('Val appear mAP:', avg_best_metrics['val_appear_mAP'])
print('Train grade mAP:', avg_best_metrics['train_grade_mAP'])
print('Val grade mAP:', avg_best_metrics['val_grade_mAP'])
  