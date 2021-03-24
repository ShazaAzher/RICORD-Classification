import torchxrayvision as xrv
import torch
#import torchvision
from torchsummary import summary
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import argparse
import os
import sys
from time import time
import utils

parser = argparse.ArgumentParser(description='Train a pretrained DenseNet on RICORD')
parser.add_argument('--weights', default='all', type=str, help="Dataset pretrained on; one of 'all', 'rsna', 'nih', 'pc', 'chex', 'mimic_nb', 'mimic_ch'")
parser.add_argument('--train_subjects', type=str, help='Text file containing subject IDs for training')
parser.add_argument('--out_type', default='both', type=str, help="Which output heads to train; one of 'appear', 'grade', 'both'")
parser.add_argument('--loss_param', default=0.5, type=float, help="Proportional of total loss that appearance classification contributes to; ignored if out_type not 'both'")
parser.add_argument('--lr', default=0.001, type=float, help="Learning rate for Adam optimizer")
parser.add_argument('--batch_size', default=32, type=int, help="Size of minibatches")
parser.add_argument('--k', default=1, type=int, help="Number of folds for cross-validation")
args = parser.parse_args()

appear_mapping, grade_mapping = utils.get_label_mappings()
im_height, im_width = 224, 224
max_pixel_value = 4096 # (12-bit)
annotation_filepath = '../data/final_annotations.csv'
stacked_images_filepath = '../data/stacked_images_resized.npy'

train_subjs = open(args.train_subjects).read().splitlines()
annots = pd.read_csv(annotation_filepath)
annots = annots.replace(np.nan, '', regex=True)
annots = annots[annots['SubjectID'].isin(train_subjs)]
annots = annots.reset_index(drop=True)
images = np.load(stacked_images_filepath)

class Network(torch.nn.Module):
    def __init__(self, weights):
        super(Network, self).__init__()
        self.base_model = xrv.models.DenseNet(weights=weights)
        num_in = self.base_model.features.norm5.bias.shape[0]
        self.base_model.classifier = torch.nn.Identity() # remove pretrained FC layer
        self.base_model.op_threshs = None # turn off output normalization
        self.appear_fc = torch.nn.Linear(in_features=num_in, out_features=len(appear_mapping))
        self.grade_fc = torch.nn.Linear(in_features=num_in, out_features=len(grade_mapping))
    
    def forward(self, input, out_type):
        x = self.base_model(input)
        
        if out_type == 'both':
            return self.appear_fc(x), self.grade_fc(x)
        elif out_type == 'appear':
            return self.appear_fc(x), None
        elif out_type == 'grade':
            return None, self.grade_fc(x)
        else:
            raise ValueError("output type must be one of 'appear', 'grade', or 'both'")

    def freeze(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, subjs):
        
        split_annots = annots[annots['SubjectID'].isin(subjs)]
        self.appear_labels = [appear_mapping[label] for label in split_annots['AppearLabel']]
        self.grade_labels = [grade_mapping[label] if label in grade_mapping else -1 for label in split_annots['GradeLabel']]
        self.images = np.take(images, split_annots.index, axis=0)
        
    def __getitem__(self, index):
        return (self.images[index], self.appear_labels[index], self.grade_labels[index])

    def __len__(self):
        return len(self.images)


def train_model(weights, train_subjs, out_type, lr=0.001, a=0.5, k=1, num_epochs=100, batch_size=32):
    
    fold_subjs, num_ims_fold = utils.fold_split(annots, train_subjs, k)
    
    appear_loss_func = torch.nn.CrossEntropyLoss()
    grade_loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    datasets = {}
    data_loaders = {}
    
    appear = (out_type in ['appear', 'both'])
    grade = (out_type in ['grade', 'both'])

    # don't weight losses if not running multi-task classification
    if out_type == 'appear': a = 1
    elif out_type == 'grade': a = 0

    # init dict for storing metrics
    history = {}
    for metric in ['appear_acc', 'grade_acc', 'appear_auroc', 'grade_auroc', 'total_loss', 'appear_loss', 'grade_loss']:
        history[metric] = {}
        for phase in ['train', 'val']:
            history[metric][phase] = [[] for _ in k]


    # for each fold in k-fold CV
    for i in range(k):
        
        # create data loaders
        datasets['val'] = Dataset(fold_subjs[i])
        data_loaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
        datasets['train'] = Dataset(np.concatenate(np.delete(fold_subjs, i)))
        data_loaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    
        # load model and optimizer
        model = Network(weights)
        model.freeze()
        trainable_params = []
        if appear: trainable_params += list(model.appear_fc.parameters())
        if grade: trainable_params += list(model.grade_fc.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr)

        # iterate over epochs
        for epoch in range(num_epochs):
            
            for phase in ['train', 'val']:
                
                # make buffers for output probabilities and targets, for epoch metric calculations
                count = 0
                running_total_loss = 0
                running_appear_loss = 0
                running_grade_loss = 0
                if appear: 
                    all_appear_prob = np.zeros(len(datasets[phase]), len(appear_mapping))
                    all_appear_targets = np.zeros(len(datasets[phase]))
                if grade:
                    all_grade_prob = np.zeros(len(datasets[phase]), len(grade_mapping))
                    all_grade_targets = np.zeros(len(datasets[phase]))

                # iterate over batches
                for images, appear_targets, grade_targets in data_loaders[phase]:
                    
                    optimizer.zero_grad()
                    
                    # inference and gradient step
                    with torch.set_grad_enabled(phase == 'train'):
                        appear_out, grade_out = model.forward(images, out_type)
                        
                        loss_appear = appear_loss_func(appear_out, appear_targets) if appear else 0
                        loss_grade = grade_loss_func(grade_out, grade_targets) if grade else 0
                        loss = a * loss_appear + (1-a) * loss_grade
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # store batch metrics
                    if appear:
                        all_appear_prob[count: count + len(images), :] = softmax(appear_out, axis=1)
                        all_appear_targets[count: count + len(images)] = appear_targets
                    if grade:
                        all_grade_prob[count: count + len(images), :] = softmax(grade_out, axis=1)
                        all_grade_targets[count: count + len(images)] = grade_targets
                    running_total_loss += loss.item() * len(images)
                    if appear: running_appear_loss += loss_appear.item() * len(images)
                    if grade: running_grade_loss += loss_grade.item() * np.sum(grade_targets != -1)
                    count += len(images)

                # store epoch metrics
                if appear:
                    auroc = roc_auc_score(all_appear_targets, all_appear_prob)
                    history['appear_auroc'][phase][i].append(auroc)
                    _, all_appear_preds = np.max(all_appear_prob, 1)
                    acc = accuracy_score(all_appear_targets, all_appear_preds)
                    history['appear_acc'][phase][i].append(acc)
                if grade:
                    # remove rows with no target grade label
                    no_target_ind = (all_grade_targets == -1)
                    all_grade_targets = np.delete(all_grade_targets, no_target_ind)
                    all_grade_prob = np.delete(all_appear_prob, no_target_ind, axis=0)

                    auroc = roc_auc_score(all_grade_targets, all_grade_prob)
                    history['grade_auroc'][phase][i].append(auroc)
                    _, all_grade_preds = np.max(all_grade_prob, 1)
                    acc = accuracy_score(all_grade_targets, all_grade_preds)
                    history['grade_acc'][phase][i].append(acc)
                history['total_loss'][phase][i].append(running_total_loss/len(datasets[phase]))
                if appear: history['appear_loss'][phase][i].append(running_appear_loss/len(datasets[phase]))
                if grade: history['grade_loss'][phase][i].append(running_grade_loss/len(datasets[phase]))


train_model(args.weights, train_subjs, args.out_type, k=args.k, lr=args.lr, a=args.loss_param, num_epochs=100)