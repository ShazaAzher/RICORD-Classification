import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, subjs, annots, images, label_mappings, augment=None):
        
        split_annots = annots[annots['SubjectID'].isin(subjs)]
        self.appear_labels = [label_mappings['appear'][label] for label in split_annots['AppearLabel']]
        self.grade_labels = [label_mappings['grade'].get(label, -1) for label in split_annots['GradeLabel']]
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