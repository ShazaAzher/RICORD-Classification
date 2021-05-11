import torch
import numpy as np
import torchvision
import sys

class Dataset(torch.utils.data.Dataset):
    def __init__(self, subjs, annots, images, label_mappings, augment=None, target_type='hard'):
        split_annots = annots[annots['SubjectID'].isin(subjs)]
        self.target_type = target_type

        # appearance labels
        appear_counts = split_annots[label_mappings['appear'].keys()]
        appear_counts = appear_counts.rename(columns=label_mappings['appear']).sort_index(axis=1)
        appear_counts = np.array(appear_counts, dtype='float')
        self.appear_probs = appear_counts/np.sum(appear_counts, axis=1)[:, None]
        self.appear_labels = np.argmax(appear_counts, axis=1)
        for i, adjud in enumerate(list(split_annots['AppearAdjudication'])):
          if adjud in label_mappings['appear']:
            self.appear_labels[i] = label_mappings['appear'][adjud]

        # grade labels
        grade_counts = split_annots[label_mappings['grade'].keys()]
        grade_counts = grade_counts.rename(columns=label_mappings['grade']).sort_index(axis=1)
        grade_counts = np.array(grade_counts, dtype='float')
        keep_rows = np.any(grade_counts, axis=1)
        self.grade_probs = np.array(grade_counts)
        self.grade_probs[keep_rows] /= np.sum(grade_counts[keep_rows], axis=1)[:, None]
        self.grade_probs[~keep_rows, :] = -1
        self.grade_labels = np.argmax(grade_counts, axis=1)
        for i, adjud in enumerate(list(split_annots['GradeAdjudication'])):
          if adjud in label_mappings['grade']:
            self.grade_labels[i] = label_mappings['grade'][adjud]
          elif self.appear_labels[i] == label_mappings['appear']['Negative for Pneumonia']:
            self.grade_labels[i] = -1
        
        # images
        self.images = np.take(images, split_annots.index, axis=0)
        if augment:
          transforms = []
          if 'horiz_flip' in augment: 
            transforms.append(torchvision.transforms.RandomHorizontalFlip(p=augment['horiz_flip']))
          rotation = augment.get('rotation', 0)
          translation = (augment['translation'], augment['translation']) if 'translation' in augment else None
          scaling = (1-augment['scaling'], 1+augment['scaling']) if 'scaling' in augment else None
          if rotation or translation or scaling:
            transforms.append(torchvision.transforms.RandomAffine(rotation, translate=translation,scale=scaling))
          self.transform = torchvision.transforms.Compose(transforms)
        else: self.transform = None
        
    def __getitem__(self, index):
        image = torch.Tensor(self.images[index])
        if self.transform: image = self.transform(image)
        appear_label = np.random.choice(4, p=self.appear_probs[index]) if self.target_type=='sample' else self.appear_labels[index]
        if (self.target_type=='sample' and self.grade_labels[index] != -1):
          grade_label = np.random.choice(3, p=self.grade_probs[index])  
        else: 
          grade_label = self.grade_labels[index]
        return (image, appear_label, self.appear_probs[index], grade_label, self.grade_probs[index])

    def __len__(self):
        return len(self.images)