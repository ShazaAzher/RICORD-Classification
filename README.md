# COVID Classification on the RICORD Dataset
This project contains code for training and testing a DenseNet model on the RICORD dataset of chest x-rays. Each scan can be classified as having 
"Typical", "Atyptical", "Indeterminate", or "Negative" appearance, as well as "Mild", "Moderate", or "Severe" disease grading. 

## Setup
First, download the image folder from the 
[Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281) and place it in a folder called *images* within 
the [data](data) folder. Note that the annotations and clinical data files have already been preprocessed and placed in [data](data), so you do not need
to download them.

Create text files corresponding to training patient and testing patient IDs. You may use the ones found at [train_subjects.txt](data/train_subjects.txt) and 
[test_subjects.txt](data/test_subjects.txt), or use your own. The [split.py](code/scripts/data_processing/split.py) script can randomly divide the patients
into train and test sets. Note that the training code uses k-fold cross validation to choose the best model, so you do not need to hold out a separate 
validation set.

Preprocesses the images by running the following script once for the training patients and testing patients:
```
python code/scripts/data_processing/preprocess_images.py TRAIN_PATIENTS_TXT_PATH TRAIN_IMAGES_SAVE_PATH
python code/scripts/data_processing/preprocess_images.py TEST_PATIENTS_TXT_PATH TEST_IMAGES_SAVE_PATH
```

## Training
To train on the preprocessed training images, run
```
python code/scripts/model/train_model.py TRAIN_PATIENTS_TXT_PATH MODEL_SAVE_PATH
```
A variety of options for training are supported:
- Loading weights from pretraining on other x-ray datasets or your own custom model
- Optimizer parameters such as learning rate
- Data augmentation techniques such as rotations and scalings
- Different losses such as "soft" cross-entropy (taking radiologist uncertainty into account) and focal loss

For more details on using these, please refer to the output of
```
python code/scripts/model/train_model.py -h
```

## Testing
To test a trained model on the preprocessed test set images, run
```
python code/scripts/model/inference.py TEST_PATIENTS_TXT_PATH MODEL_LOAD_PATH
```
This will print out mAP, AUC, and accuracy scores. Scripts to help with more fine-grained analysis can be found in the [metrics](code/scripts/metrics) folder.
