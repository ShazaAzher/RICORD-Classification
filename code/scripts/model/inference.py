import torch
import torchvision
from torchsummary import summary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import argparse
import os
import glob
import sys
from time import time
import utils
from network import Network, set_bn_eval
from dataset import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(0)

parser = argparse.ArgumentParser(description='Compute predictions')
parser.add_argument('test_subjects', type=str, help='Text file containing subject IDs for testing')
parser.add_argument('model_folder', type=str, help='Path of model')
#parser.add_argument('pred_save_path', type=str, help="Where to save preds")
args = parser.parse_args()

label_mappings = utils.get_label_mappings()
annotation_filepath = '../data/final_annotations.csv'
stacked_images_filepath = '../data/test_images_preprocessed.npy'

test_subjs = open(args.test_subjects).read().splitlines()
annots = pd.read_csv(annotation_filepath)
annots = annots.replace(np.nan, '', regex=True)
annots = annots[annots['SubjectID'].isin(test_subjs)]
annots = annots.reset_index(drop=True)
all_images = np.load(stacked_images_filepath)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size=32
dataset = Dataset(test_subjs, annots, all_images, label_mappings)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
len_dataset = len(dataset)

best_model = None
overall_best_loss = float('inf')
for subfolder in os.listdir(args.model_folder):
  if not len(glob.glob(os.path.join(args.model_folder, subfolder, '*best*'))): continue
  metrics = pd.read_csv(os.path.join(args.model_folder, subfolder, 'history.csv'), index_col=0)
  best_loss = metrics['val_total_loss'].idxmin()
  if best_loss < overall_best_loss:
    best_loss = overall_best_loss
    best_model = glob.glob(os.path.join(args.model_folder, subfolder, '*best*'))[0]
  

checkpoint = torch.load(best_model)
model = Network()
model.load_state_dict(checkpoint['state_dict'])
model.freeze()
model.eval()
model = model.to(device)


all_appear_prob = torch.zeros([len_dataset, 4])
all_appear_targets = torch.zeros(len_dataset)
all_grade_prob = torch.zeros([len_dataset, 3])
all_grade_targets = torch.zeros(len_dataset)
count = 0

for batch in data_loader:
  images, appear_targets, appear_probs, grade_targets, grade_probs = batch

  images = images.to(device)

  with torch.no_grad():
    appear_out, grade_out = model(images, 'both')

    all_appear_prob[count: count + len(images), :] = torch.nn.functional.softmax(appear_out.detach(), dim=1)
    all_appear_targets[count: count + len(images)] = appear_targets
    all_grade_prob[count: count + len(images), :] = torch.nn.functional.softmax(grade_out.detach(), dim=1)
    all_grade_targets[count: count + len(images)] = grade_targets

  count += len(images)

APs = utils.multiclass_AP(all_appear_targets, all_appear_prob)
print("Appear mAP:", np.mean(APs))
auc = roc_auc_score(all_appear_targets, all_appear_prob, multi_class='ovr')
print("Appear AUC:", auc)
all_appear_preds = np.argmax(all_appear_prob, 1)
acc = accuracy_score(all_appear_targets, all_appear_preds)
print("Appear Accuracy:", acc)

mask = (all_grade_targets != -1)
all_grade_targets = all_grade_targets[mask]
all_grade_prob = all_grade_prob[mask, :]
APs = utils.multiclass_AP(all_grade_targets, all_grade_prob)
print("Grade mAP:", np.mean(APs))
auc = roc_auc_score(all_grade_targets, all_grade_prob, multi_class='ovr')
print("Grade AUC:", auc)
all_grade_preds = np.argmax(all_grade_prob, 1)
acc = accuracy_score(all_grade_targets, all_grade_preds)
print("Grade Accuracy:", acc)
