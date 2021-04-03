import pandas as pd
import numpy as np
import argparse
import pydicom
import torchxrayvision as xrv
import torchvision
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Preprocess images and save as array')
parser.add_argument('subjects', type=str, help='Text file containing subject IDs')
parser.add_argument('save_path', type=str, help='File path to save image array')
args = parser.parse_args()

annotation_filepath = '../data/final_annotations.csv'
im_height, im_width = 224, 224

# def preprocess_image(self, im):
#     # center crop a square
#     if im.shape[0] > im.shape[1]:
#         crop_width = (im.shape[0] - im.shape[1])//2
#         im = im[crop_width : im.shape[1]+crop_width, :]
#     if im.shape[1] > im.shape[0]:
#         crop_width = (im.shape[1] - im.shape[0])//2
#         im = im[:, crop_width : im.shape[0]+crop_width]
    
#     im = np.interp(im, (np.min(im), np.max(im)), (-1024, 1024)) # scale to [-1024, 1024]
#     im = np.array(Image.fromarray(im).resize((im_height, im_width))) # resize to 224x224
#     return im

def preprocess_image(im):

    max_val = np.max(im)
    max_bound = 256
    while True:
        if max_val < max_bound: break
        max_bound *= 2
    
    im = np.expand_dims(im, 0)
    im = xrv.datasets.normalize(im, max_bound)
    crop_and_resize = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), 
                                                     xrv.datasets.XRayResizer(im_height)])
    im = crop_and_resize(im)
    # plt.imshow(np.squeeze(im), cmap='gray')
    # plt.show()
    return im

subjs = open(args.subjects).read().splitlines()
annots = pd.read_csv(annotation_filepath)
annots = annots[annots['SubjectID'].isin(subjs)]

images = np.zeros([len(annots), 1, im_height, im_width], dtype=np.float32)
for i, filepath in enumerate(list(annots['FilePath'])):
    im = pydicom.dcmread(filepath).pixel_array
    im = preprocess_image(im)
    images[i,:,:,:] = im
    
np.save(args.save_path, images)