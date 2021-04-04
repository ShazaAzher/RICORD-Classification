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
from network import Network
from dataset import Dataset
from metric_logger import Metric_Logger

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Train a pretrained DenseNet on RICORD')
parser.add_argument('train_subjects', type=str, help='Text file containing subject IDs for training')
parser.add_argument('model_save_path', type=str, help='Path of folder where weights and history will be saved')
parser.add_argument('--weights', default='all', type=str, help="Dataset pretrained on; one of 'all', 'rsna', 'nih', 'pc', 'chex', 'mimic_nb', 'mimic_ch'")
parser.add_argument('--out_type', default='both', type=str, help="Which output heads to train; one of 'appear', 'grade', 'both'")
parser.add_argument('--loss_param', default=0.5, type=float, help="Proportional of total loss that appearance classification contributes to; ignored if out_type not 'both'")
parser.add_argument('--lr', default=0.001, type=float, help="Learning rate for Adam optimizer")
parser.add_argument('--lr_factor', default=None, type=float, help="Steps down LR after 'patience' epochs without improvement")
parser.add_argument('--batch_size', default=32, type=int, help="Size of minibatches")
parser.add_argument('--k', default=1, type=int, help="Number of folds for cross-validation")
parser.add_argument('--rotation', default=None, type=float, help="Degrees of rotation as part of augmentation")
parser.add_argument('--translation', default=None, type=float, help="Fraction of translation as part of augmentation")
parser.add_argument('--scaling', default=None, type=float, help="Fraction of scaling as part of augmentation")
parser.add_argument('--horiz_flip_prob', default=None, type=float, help="Probability of horizontal flip as part of augmentation")
parser.add_argument('--dropout', default=None, type=float, help="Probability of dropout after the last convolutional layer")
parser.add_argument('--finetune_path', default=None, type=str, help="If path provided, overrides 'weights' arg and loads and finetunes model")
args = parser.parse_args()


def train_model(train_subjs, annots, all_images, label_mappings, save_path, out_type,
                weights=None, finetune_path=None, lr=0.001, lr_factor=None, 
                a=0.5, k=1, num_epochs=100, batch_size=32, patience=10, 
                augment=None, dropout_prob=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    fold_subjs, num_ims_fold = utils.fold_split(annots, train_subjs, k)
    phases = ['train', 'val'] if k > 1 else ['train']
    
    appear_loss_func = torch.nn.CrossEntropyLoss()
    grade_loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    datasets = {}
    data_loaders = {}
    
    appear = (out_type in ['appear', 'both'])
    grade = (out_type in ['grade', 'both'])

    # don't weight losses if not running multi-task classification
    if out_type == 'appear': a = 1
    elif out_type == 'grade': a = 0

    # metrics to track
    logger = Metric_Logger(appear, grade)

    # for each fold in k-fold CV
    for fold in range(k):
        
        print('\nStarting fold', fold+1)

        logger.init_fold(fold)
        fold_path = os.path.join(save_path, 'fold_' + str(fold+1))
        if not os.path.exists(fold_path): os.mkdir(fold_path)
        num_epochs_no_imp = 0
        stop_flag = False # whether to terminate early
        imp_on_lr = False # whether any improvement was seen with the current LR
        
        # create data loaders
        datasets['train'] = Dataset(np.concatenate(np.delete(fold_subjs, fold)), annots, 
                                    all_images, label_mappings, augment=augment)
        data_loaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
        if k > 1:
            datasets['val'] = Dataset(fold_subjs[fold], annots, all_images, label_mappings)
            data_loaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
    
        # setup model and optimizer
        if finetune_path:
          finetune_fold_path = glob.glob(os.path.join(finetune_path, 'fold_'+str(fold+1), '*best*'))[0]
          checkpoint = torch.load(finetune_fold_path)
          model = Network(dropout_prob=dropout_prob)
          model.load_state_dict(checkpoint['state_dict'])
          model.unfreeze()
          trainable_params = [param for param in model.parameters() if param.requires_grad]
          if appear: trainable_params += list(model.appear_fc.parameters())
          if grade: trainable_params += list(model.grade_fc.parameters())
          optimizer = torch.optim.Adam(trainable_params, lr=1e-5)
          best_val_loss = utils.get_best_val_loss(finetune_path, fold+1)
        else:
          model = Network(weights=weights, dropout_prob=dropout_prob)
          model.freeze()
          trainable_params = []
          if appear: trainable_params += list(model.appear_fc.parameters())
          if grade: trainable_params += list(model.grade_fc.parameters())
          optimizer = torch.optim.Adam(trainable_params, lr=lr)
          best_val_loss = float('inf')
        model = model.to(device)
        

        # iterate over epochs
        for epoch in range(num_epochs):
            
            logger.init_epoch(epoch)

            for phase in phases:
                
                # set to train or eval mode
                model.train(mode = (phase == 'train'))

                # make buffers for output probabilities and targets, for epoch metric calculations
                logger.init_phase(phase, len(datasets[phase]))

                # iterate over batches
                for images, appear_targets, grade_targets in data_loaders[phase]:
                    
                    images = images.to(device)
                    if appear: appear_targets = appear_targets.to(device)
                    if grade: grade_targets = grade_targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    # inference and gradient step
                    with torch.set_grad_enabled(phase == 'train'):
                        appear_out, grade_out = model(images, out_type)
                        
                        loss_appear = appear_loss_func(appear_out, appear_targets) if appear else 0
                        loss_grade = grade_loss_func(grade_out, grade_targets) if grade else 0
                        loss = a * loss_appear + (1-a) * loss_grade
                        logger.save_losses(loss_appear, loss_grade, loss)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # store batch metrics
                    logger.batch_metrics(appear_out, appear_targets, grade_out, grade_targets)

                # store epoch metrics and get average total loss for phase epoch
                avg_loss = logger.phase_metrics()

                # model saving, LR stepping, and early stopping
                if phase == 'val':
                    state = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    if epoch > 0: os.remove(glob.glob(os.path.join(fold_path, 'model_latest*'))[0])
                    torch.save(state, os.path.join(fold_path, 'model_latest_ep{}_loss{:.2f}.pt'.format(epoch+1, avg_loss)))
                    if avg_loss < best_val_loss:
                        logger.save_best(fold_path)
                        prev_best = glob.glob(os.path.join(fold_path, 'model_best*'))
                        if prev_best: os.remove(prev_best[0])
                        torch.save(state, os.path.join(fold_path, 'model_best_ep{}_loss{:.2f}.pt'.format(epoch+1, avg_loss)))
                        best_val_loss = avg_loss
                        num_epochs_no_imp = 0
                        imp_on_lr = True
                    else:
                        num_epochs_no_imp += 1
                        if num_epochs_no_imp >= patience:
                            if imp_on_lr and lr_factor:
                                for g in optimizer.param_groups: g['lr'] *= lr_factor
                                num_epochs_no_imp = 0
                                imp_on_lr = False
                                print("\nSTEPPING DOWN LR")
                            else:
                                stop_flag = True 
            
            # save and print metrics
            logger.epoch_metrics(fold_path)
            if stop_flag: 
                print("STOPPING EARLY\n")
                break


label_mappings = utils.get_label_mappings()
annotation_filepath = '../data/final_annotations.csv'
stacked_images_filepath = '../data/preprocessed_train_images.npy'

augment = {}
if args.rotation: augment['rotation'] = args.rotation
if args.translation: augment['translation'] = args.translation
if args.scaling: augment['scaling'] = args.scaling
if args.horiz_flip_prob: augment['horiz_flip'] = args.horiz_flip_prob

train_subjs = open(args.train_subjects).read().splitlines()
annots = pd.read_csv(annotation_filepath)
annots = annots.replace(np.nan, '', regex=True)
annots = annots[annots['SubjectID'].isin(train_subjs)]
annots = annots.reset_index(drop=True)
all_images = np.load(stacked_images_filepath)

train_model(train_subjs, annots, all_images, label_mappings, args.model_save_path, 
            args.out_type, weights=args.weights, finetune_path=args.finetune_path, 
            k=args.k, lr=args.lr, lr_factor=args.lr_factor, a=args.loss_param, 
            num_epochs=200, augment=augment, dropout_prob=args.dropout)
          