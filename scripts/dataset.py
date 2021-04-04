import torch
import numpy as np
import torchvision

class Dataset(torch.utils.data.Dataset):
    def __init__(self, subjs, annots, images, label_mappings, augment=None):
        
        split_annots = annots[annots['SubjectID'].isin(subjs)]
        self.appear_labels = [label_mappings['appear'][label] for label in split_annots['AppearLabel']]
        self.grade_labels = [label_mappings['grade'].get(label, -1) for label in split_annots['GradeLabel']]
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
        return (image, self.appear_labels[index], self.grade_labels[index])

    def __len__(self):
        return len(self.images)