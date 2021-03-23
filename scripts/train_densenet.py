import torchxrayvision as xrv
import torch
#import torchvision
from torchsummary import summary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import argparse
import os
import sys
from time import time
from utils import get_label_mappings

parser = argparse.ArgumentParser(description='Train a pretrained DenseNet on RICORD')
parser.add_argument('--weights', default='all', type=str, help="Dataset pretrained on; one of 'all', 'rsna', 'nih', 'pc', 'chex', 'mimic_nb', 'mimic_ch'")
parser.add_argument('--train_subjects', type=str, help='Text file containing subject IDs for training')
parser.add_argument('--out_type', default='both', type=str, help="Which output heads to train; one of 'appear', 'grade', 'both'")
args = parser.parse_args()

appear_mapping, grade_mapping = get_label_mappings()
im_height, im_width = 224, 224
max_pixel_value = 4096 # (12-bit)
annotation_filepath = '../data/final_annotations.csv'
stacked_images_filepath = '../data/stacked_images_resized.npy'

class RicordNetwork(torch.nn.Module):
    def __init__(self, weights):
        super(RicordNetwork, self).__init__()
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
            return self.appear_fc(x), 
        elif out_type == 'grade':
            return self.grade_fc(x)
        else:
            raise ValueError("output type must be one of 'appear', 'grade', or 'both'")

    def freeze(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

class RicordDataset(torch.utils.data.Dataset):
    def __init__(self, train_subj_file):
        train_subjs = open(train_subj_file).read().splitlines()
        annots = pd.read_csv(annotation_filepath)
        annots = annots.replace(np.nan, '', regex=True)
        annots = annots[annots['SubjectID'].isin(train_subjs)]

        self.appear_labels = [appear_mapping[label] for label in annots['AppearLabel']]
        self.grade_labels = [grade_mapping[label] if label in grade_mapping else -1 for label in annots['GradeLabel']]

        if not os.path.exists(stacked_images_filepath):
            self.images = np.zeros([len(annots), 1, im_height, im_width], dtype=np.float32)
            for i, filepath in enumerate(list(annots['FilePath'])):
                im = pydicom.dcmread(filepath)
                im = self._preprocess_image(im.pixel_array)
                im = torch.unsqueeze(torch.Tensor(im), 0)
                self.images[i,:,:,:] = im
            np.save(stacked_images_filepath, self.images)
        else:
            self.images = np.load(stacked_images_filepath)


    def __getitem__(self, index):
        return (self.images[index], self.appear_labels[index], self.grade_labels[index])


    def __len__(self):
        return len(self.appear_labels)


    def _preprocess_image(self, im):
        # center crop a square
        if im.shape[0] > im.shape[1]:
            crop_width = (im.shape[0] - im.shape[1])//2
            im = im[crop_width : im.shape[1]+crop_width, :]
        if im.shape[1] > im.shape[0]:
            crop_width = (im.shape[1] - im.shape[0])//2
            im = im[:, crop_width : im.shape[0]+crop_width]
        
        im = np.interp(im, (np.min(im), np.max(im)), (-1024, 1024)) # scale to [-1024, 1024]
        im = np.array(Image.fromarray(im).resize((im_height, im_width))) # resize to 224x224
        return im

def train_model(model, data_loader, out_type, num_epochs=100):

    trainable_params = []
    if out_type in ['appear', 'both']: trainable_params += list(model.appear_fc.parameters())
    if out_type in ['grade', 'both']: trainable_params += list(model.grade_fc.parameters())
    optimizer = torch.optim.Adam(trainable_params)

    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('\nEpoch', epoch)
        for images, appear_labels, grade_labels in data_loader:
            optimizer.zero_grad()
            outputs = model.forward(images, out_type)
            _, preds = torch.max(outputs, 1)
            loss = loss_func(outputs, appear_labels)
            loss.backward()
            optimizer.step()
            acc = (appear_labels == preds).sum().item()/len(preds)*100
            print(loss.item(), acc)


batch_size = 32
dataset = RicordDataset(args.train_subjects)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = RicordNetwork(args.weights)
model.freeze()

train_model(model, data_loader, args.out_type, num_epochs=10)