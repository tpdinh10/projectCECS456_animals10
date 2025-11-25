#CECS 456 Animals10 Image Classification

This project trains a compact convolutional neural network to classify ten animal categories using the Animals10 dataset. It is created for the CECS 456 machine learning course and includes a complete workflow from dataset preparation to training, evaluation, and reporting.

## Folder structure

```text
project456/
  data/
    animals10/              # original Kaggle Animals10 dataset (10 folders)
  src/
    train.py                # main training script (fast version)
  results/
    animals10_cnn.h5        # saved CNN model
    confusion_matrix.png    # confusion matrix from validation set
    training_curves.png     # accuracy and loss plots
    history.json            # saved training history
  notebooks/
    EDA.ipynb               # optional: data exploration
    training_results.ipynb  # optional: visualization of training results
  report/
    report.pdf              # final project report
  requirements.txt          # required packages
```

## How to run


How to run

1. Install dependencies

pip install -r requirements.txt


2. Place the Animals10 dataset

Download the dataset from Kaggle and place all ten class folders inside:

project456/data/animals10/


The folders should have the original Italian class names from the dataset.

3. Train the CNN

python src/train.py


This script trains a lightweight CNN using
input size 64 by 64
ten epochs
two convolution blocks
a dense layer with sixty four units
fifteen percent validation split

Training runs quickly and is suitable for laptop CPUs.

4. View results

After training, the following files appear in the results folder:

animals10_cnn.h5

confusion_matrix.png

training_curves.png

history.json

## Notes

This project now uses the Kaggle Animals10 dataset.

The CNN architecture is intentionally small to keep training time under a few minutes.
