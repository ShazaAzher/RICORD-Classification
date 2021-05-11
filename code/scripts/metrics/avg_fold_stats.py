import pandas as pd 
import numpy as np 
import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Print average metrics over folds')
parser.add_argument('model_folder', type=str, help='Folder containing subfolders for each fold')
parser.add_argument('--fallback_folder', type=str, help='Folder to use if best model doesn\'t exist (finetuning case)')
args = parser.parse_args()

all_best_metrics = []
for subfolder in os.listdir(args.model_folder):
  model_folder = args.model_folder if len(glob.glob(os.path.join(args.model_folder, subfolder, '*best*'))) \
    else args.fallback_folder
  metrics = pd.read_csv(os.path.join(model_folder, subfolder, 'history.csv'), index_col=0)
  best_metrics = metrics.iloc[metrics['val_total_loss'].idxmin()]
  all_best_metrics.append(best_metrics)

all_best_metrics = pd.DataFrame(all_best_metrics)
avg_best_metrics = all_best_metrics.mean(axis=0)
#print('Train appear mAP:', avg_best_metrics['train_appear_mAP'])
print('Val appear mAP:', avg_best_metrics['val_appear_mAP'])
#print('Train grade mAP:', avg_best_metrics['train_grade_mAP'])
print('Val grade mAP:', avg_best_metrics['val_grade_mAP'])

print('Val appear AP c0:', avg_best_metrics['val_appear_AP_c0'])
print('Val appear AP c1:', avg_best_metrics['val_appear_AP_c1'])
print('Val appear AP c2:', avg_best_metrics['val_appear_AP_c2'])
print('Val appear AP c3:', avg_best_metrics['val_appear_AP_c3'])

print('Val grade AP c0:', avg_best_metrics['val_grade_AP_c0'])
print('Val grade AP c1:', avg_best_metrics['val_grade_AP_c1'])
print('Val grade AP c2:', avg_best_metrics['val_grade_AP_c2'])

# print()
# print(all_best_metrics[['val_appear_mAP', 'val_grade_mAP']])

# print('Val appear acc:', avg_best_metrics['val_appear_acc'])
# print('Val grade acc:', avg_best_metrics['val_grade_acc'])
# print('Val appear auc:', avg_best_metrics['val_appear_auroc'])
# print('Val grade auc:', avg_best_metrics['val_grade_auroc'])
  