import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import utils
from time import time
import os

class Metric_Logger():
  def __init__(self, appear, grade):
    self.appear = appear
    self.grade = grade
    self.cols = ['appear_acc', 'appear_auroc', 'appear_AP_c0', 'appear_AP_c1', 'appear_AP_c2', 'appear_AP_c3', 'appear_mAP',
            'grade_acc', 'grade_auroc', 'grade_AP_c0', 'grade_AP_c1', 'grade_AP_c2', 'grade_mAP',
            'total_loss', 'appear_loss', 'grade_loss']
    self.cols = ['epoch'] + ['train_' + col for col in self.cols] +  ['val_' + col for col in self.cols] + ['time']
    self.print_metrics = ['epoch', 'train_total_loss', 'val_total_loss']
    if self.appear: self.print_metrics += ['train_appear_mAP', 'val_appear_mAP', 'train_appear_loss','val_appear_loss']
    if self.grade: self.print_metrics += ['train_grade_mAP', 'val_grade_mAP', 'train_grade_loss', 'val_grade_loss']

  def init_fold(self, fold):
    self.fold = fold
    self.history = pd.DataFrame(columns = self.cols)

  def init_epoch(self, epoch):
    self.epoch_stats = {'epoch': epoch+1}
    self.start_time = time()

  def init_phase(self, phase, len_dataset):
    self.count = 0
    self.running_total_loss = 0
    self.running_appear_loss = 0
    self.running_grade_loss = 0
    self.phase = phase
    self.len_dataset = len_dataset
    if self.appear: 
        self.all_appear_prob = torch.zeros([len_dataset, 4])
        self.all_appear_targets = torch.zeros(len_dataset)
    if self.grade:
        self.all_grade_prob = torch.zeros([len_dataset, 3])
        self.all_grade_targets = torch.zeros(len_dataset)
  
  def save_losses(self, appear_loss, grade_loss, loss):
    self.appear_loss = appear_loss
    self.grade_loss = grade_loss
    self.loss = loss

  def batch_metrics(self, appear_out, appear_targets, grade_out, grade_targets):
    batch_size = len(appear_targets)
    if self.appear:
        self.all_appear_prob[self.count: self.count + batch_size, :] = torch.nn.functional.softmax(appear_out.detach(), dim=1)
        self.all_appear_targets[self.count: self.count + batch_size] = appear_targets
        self.running_appear_loss += self.appear_loss.item() * batch_size
    if self.grade:
        self.all_grade_prob[self.count: self.count + batch_size, :] = torch.nn.functional.softmax(grade_out.detach(), dim=1)
        self.all_grade_targets[self.count: self.count + batch_size] = grade_targets
        self.running_grade_loss += self.grade_loss.item() * torch.sum((grade_targets != -1)).item()
    self.running_total_loss += self.loss.item() * batch_size
    self.count += batch_size

  def phase_metrics(self):
    self.avg_loss = self.running_total_loss/self.len_dataset
    self.epoch_stats[self.phase + '_total_loss'] = self.avg_loss
    if self.appear:
        APs = utils.multiclass_AP(self.all_appear_targets, self.all_appear_prob)
        for i, ap in enumerate(APs):
            self.epoch_stats[self.phase + '_appear_AP_c' + str(i)] = ap
        self.epoch_stats[self.phase + '_appear_mAP'] = np.mean(APs)
        self.epoch_stats[self.phase + '_appear_auroc'] = roc_auc_score(self.all_appear_targets, self.all_appear_prob, multi_class='ovr')  
        all_appear_preds = np.argmax(self.all_appear_prob, 1)
        acc = accuracy_score(self.all_appear_targets, all_appear_preds)
        self.epoch_stats[self.phase + '_appear_acc'] = acc
        self.epoch_stats[self.phase + '_appear_loss'] = self.running_appear_loss/self.len_dataset
    if self.grade:
        # remove rows with no target grade label
        no_target_ind = (self.all_grade_targets == -1)
        self.all_grade_targets = np.delete(self.all_grade_targets, no_target_ind)
        self.all_grade_prob = np.delete(self.all_grade_prob, no_target_ind, axis=0)

        APs = utils.multiclass_AP(self.all_grade_targets, self.all_grade_prob)
        for i, ap in enumerate(APs):
            self.epoch_stats[self.phase + '_grade_AP_c' + str(i)] = ap
        self.epoch_stats[self.phase + '_grade_mAP'] = np.mean(APs)
        self.epoch_stats[self.phase + '_grade_auroc'] = roc_auc_score(self.all_grade_targets, self.all_grade_prob, multi_class='ovr') 
        all_grade_preds = np.argmax(self.all_grade_prob, 1)
        acc = accuracy_score(self.all_grade_targets, all_grade_preds)
        self.epoch_stats[self.phase + '_grade_acc'] = acc
        self.epoch_stats[self.phase + '_grade_loss'] = self.running_grade_loss/torch.sum((self.all_grade_targets != -1)).item()
    
    return self.avg_loss

  def epoch_metrics(self, fold_path):
    self.epoch_stats['time'] = time() - self.start_time
    print(pd.DataFrame(self.epoch_stats, index=[0])[self.print_metrics])
    self.history = self.history.append(self.epoch_stats, ignore_index=True)
    self.history.to_csv(os.path.join(fold_path, 'history.csv'))

  def save_best(self, fold_path):
    if self.appear:
      np.save(os.path.join(fold_path, 'appear_probs.npy'), self.all_appear_prob)
      np.save(os.path.join(fold_path, 'appear_targets.npy'), self.all_appear_targets)
    if self.grade:
      np.save(os.path.join(fold_path, 'grade_probs.npy'), self.all_grade_prob)
      np.save(os.path.join(fold_path, 'grade_targets.npy'), self.all_grade_targets)

    