import torchxrayvision as xrv
import torch
import torchvision
from torchsummary import summary
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import argparse
import os
import glob
import sys
from time import time
import utils

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
parser.add_argument('--finetune_path', default=None, type=str, help="If path provided, will load and finetune model")
args = parser.parse_args()

appear_mapping, grade_mapping = utils.get_label_mappings()
annotation_filepath = '../data/final_annotations.csv'
stacked_images_filepath = '../data/preprocessed_train_images.npy'

augment = {}
if args.rotation: augment['rotation'] = args.rotation
if args.translation: augment['translation'] = args.translation
if args.scaling: augment['scaling'] = args.scaling

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
    def __init__(self, subjs, augment=None):
        
        split_annots = annots[annots['SubjectID'].isin(subjs)]
        self.appear_labels = [appear_mapping[label] for label in split_annots['AppearLabel']]
        self.grade_labels = [grade_mapping[label] if label in grade_mapping else -1 for label in split_annots['GradeLabel']]
        self.images = np.take(images, split_annots.index, axis=0)
        if augment:
          rotation = augment.get('rotation', 0)
          translation = (augment['translation'], augment['translation']) if 'translation' in augment else None
          scaling = (1-augment['scaling'], 1+augment['scaling']) if 'scaling' in augment else None
          self.transform = torchvision.transforms.Compose( \
            [torchvision.transforms.RandomAffine(rotation, translate=translation,scale=scaling)])
        else: self.transform = None
        
    def __getitem__(self, index):
        image = torch.Tensor(self.images[index])
        if self.transform: image = self.transform(image)
        return (image, self.appear_labels[index], self.grade_labels[index])

    def __len__(self):
        return len(self.images)


def train_model(weights, train_subjs, save_path, out_type, lr=0.001, lr_factor=None, 
                a=0.5, k=1, num_epochs=100, batch_size=32, patience=10, augment=None, finetune_path=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    fold_subjs, num_ims_fold = utils.fold_split(annots, train_subjs, k)
    phases = ['train', 'val'] if k > 1 else ['train']
    
    appear_loss_func = torch.nn.CrossEntropyLoss()
    grade_loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    softmax = torch.nn.Softmax(dim=1)
    
    datasets = {}
    data_loaders = {}
    
    appear = (out_type in ['appear', 'both'])
    grade = (out_type in ['grade', 'both'])

    # don't weight losses if not running multi-task classification
    if out_type == 'appear': a = 1
    elif out_type == 'grade': a = 0

    # metrics to track
    cols = ['appear_acc', 'appear_auroc', 'appear_AP_c0', 'appear_AP_c1', 'appear_AP_c2', 'appear_AP_c3', 'appear_mAP',
            'grade_acc', 'grade_auroc', 'grade_AP_c0', 'grade_AP_c1', 'grade_AP_c2', 'grade_mAP',
            'total_loss', 'appear_loss', 'grade_loss']
    cols = ['epoch'] + ['train_' + col for col in cols] +  ['val_' + col for col in cols] + ['time']
    print_metrics = ['epoch', 'train_total_loss', 'val_total_loss']
    if appear: print_metrics += ['train_appear_mAP', 'val_appear_mAP', 'train_appear_loss','val_appear_loss']
    if grade: print_metrics += ['train_grade_mAP', 'val_grade_mAP', 'train_grade_loss', 'val_grade_loss']
    fold_mAPs = {'train_appear': np.zeros(k), 'val_appear': np.zeros(k), 'train_grade': np.zeros(k), 'val_grade': np.zeros(k)}

    # for each fold in k-fold CV
    for fold in range(k):
        
        print('\nStarting fold', fold+1)

        history = pd.DataFrame(columns = cols)
        fold_path = os.path.join(save_path, 'fold_' + str(fold+1))
        if not os.path.exists(fold_path): os.mkdir(fold_path)
        best_val_loss = float('inf')
        num_epochs_no_imp = 0
        stop_flag = False # whether to terminate early
        imp_on_lr = False # whether any improvement was seen with the current LR
        
        # create data loaders
        datasets['train'] = Dataset(np.concatenate(np.delete(fold_subjs, fold)), augment=augment)
        data_loaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
        if k > 1:
            datasets['val'] = Dataset(fold_subjs[fold])
            data_loaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
    
        # load model and optimizer
        model = Network(weights)
        model.freeze()
        model = model.to(device)
        trainable_params = []
        if appear: trainable_params += list(model.appear_fc.parameters())
        if grade: trainable_params += list(model.grade_fc.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr)

        # iterate over epochs
        for epoch in range(num_epochs):
            
            epoch_stats = {'epoch': epoch+1}
            start_time = time()

            for phase in phases:
                
                # set to train or eval mode
                model.train(mode = (phase == 'train'))

                # make buffers for output probabilities and targets, for epoch metric calculations
                count = 0
                running_total_loss = 0
                running_appear_loss = 0
                running_grade_loss = 0
                if appear: 
                    all_appear_prob = torch.zeros([len(datasets[phase]), len(appear_mapping)])
                    all_appear_targets = torch.zeros(len(datasets[phase]))
                if grade:
                    all_grade_prob = torch.zeros([len(datasets[phase]), len(grade_mapping)])
                    all_grade_targets = torch.zeros(len(datasets[phase]))

                # iterate over batches
                for images, appear_targets, grade_targets in data_loaders[phase]:
                    
                    images = images.to(device)
                    if appear: appear_targets = appear_targets.to(device)
                    if grade: grade_targets = grade_targets.to(device)
                    
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
                        all_appear_prob[count: count + len(images), :] = softmax(appear_out.detach())
                        all_appear_targets[count: count + len(images)] = appear_targets
                        running_appear_loss += loss_appear.item() * len(images)
                    if grade:
                        all_grade_prob[count: count + len(images), :] = softmax(grade_out.detach())
                        all_grade_targets[count: count + len(images)] = grade_targets
                        running_grade_loss += loss_grade.item() * torch.sum((grade_targets != -1)).item()
                    running_total_loss += loss.item() * len(images)
                    count += len(images)

                # store epoch metrics
                avg_loss = running_total_loss/len(datasets[phase])
                epoch_stats[phase + '_total_loss'] = avg_loss
                if appear:
                    APs = utils.multiclass_AP(all_appear_targets, all_appear_prob)
                    for i, ap in enumerate(APs):
                        epoch_stats[phase + '_appear_AP_c' + str(i)] = ap
                    epoch_stats[phase + '_appear_mAP'] = np.mean(APs)
                    epoch_stats[phase + '_appear_auroc'] = roc_auc_score(all_appear_targets, all_appear_prob, multi_class='ovr')
                    if avg_loss < best_val_loss: fold_mAPs[phase + '_appear'][fold] = np.mean(APs)  
                    all_appear_preds = np.argmax(all_appear_prob, 1)
                    acc = accuracy_score(all_appear_targets, all_appear_preds)
                    epoch_stats[phase + '_appear_acc'] = acc
                    epoch_stats[phase + '_appear_loss'] = running_appear_loss/len(datasets[phase])
                if grade:
                    # remove rows with no target grade label
                    no_target_ind = (all_grade_targets == -1)
                    all_grade_targets = np.delete(all_grade_targets, no_target_ind)
                    all_grade_prob = np.delete(all_grade_prob, no_target_ind, axis=0)

                    APs = utils.multiclass_AP(all_grade_targets, all_grade_prob)
                    for i, ap in enumerate(APs):
                        epoch_stats[phase + '_grade_AP_c' + str(i)] = ap
                    epoch_stats[phase + '_grade_mAP'] = np.mean(APs)
                    epoch_stats[phase + '_grade_auroc'] = roc_auc_score(all_grade_targets, all_grade_prob, multi_class='ovr')
                    if avg_loss < best_val_loss: fold_mAPs[phase + '_grade'][fold] = np.mean(APs) 
                    all_grade_preds = np.argmax(all_grade_prob, 1)
                    acc = accuracy_score(all_grade_targets, all_grade_preds)
                    epoch_stats[phase + '_grade_acc'] = acc
                    epoch_stats[phase + '_grade_loss'] = running_grade_loss/torch.sum((all_grade_targets != -1)).item()


                # model saving, LR stepping, and early stopping
                if phase == 'val':
                    state = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    if epoch > 0: os.remove(glob.glob(os.path.join(fold_path, 'model_latest*'))[0])
                    torch.save(state, os.path.join(fold_path, 'model_latest_ep{}_loss{:.2f}.pt'.format(epoch+1, avg_loss)))
                    if avg_loss < best_val_loss:
                        if epoch > 0: os.remove(glob.glob(os.path.join(fold_path, 'model_best*'))[0])
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
                    
            epoch_stats['time'] = time() - start_time
            print(pd.DataFrame(epoch_stats, index=[0])[print_metrics])
            history = history.append(epoch_stats, ignore_index=True)
            history.to_csv(os.path.join(fold_path, 'history.csv'))
            if stop_flag: 
                print("STOPPING EARLY\n")
                break
          
    for metric, vals in fold_mAPs.items():
      print(metric, 'mAP:', np.mean(vals))    


train_model(args.weights, train_subjs, args.model_save_path, args.out_type, k=args.k, 
            lr=args.lr, lr_factor=args.lr_factor, a=args.loss_param, num_epochs=200, 
            augment=augment, finetune_path=args.finetune_path)