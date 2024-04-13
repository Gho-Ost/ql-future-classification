# ql-future-solution

Our solution for the QL future hackathon 2024 problem

## Team: AI've Got Genes

## Problem description:

## Data:

Image recognition dataset:

Breast cancer is one of the most common causes of death among women worldwide. Early detection helps in reducing the number of early deaths. The data reviews the medical images of breast cancer using ultrasound scan. Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images. Breast ultrasound images can produce great results in classification, detection, and segmentation of breast cancer when combined with machine learning.

The data collected at baseline include breast ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018. The number of patients is 600 female patients. The dataset consists of 780 images with an average image size of 500*500 pixels. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes, which are normal, benign, and malignant.


## drive:
https://drive.google.com/drive/folders/1ljjVvm4S7X6dXdtbS4wCe1CKylzP5UDz?usp=sharing

## File structure:
```
.
└── data/
    ├── benign/
    │   ├── benign (1).png
    │   ├── benign (1)_mask.png
    │   ├── benign (2).png
    │   ├── benign (2)_mask.png
    │   └── ...
    ├── malignant/
    │   ├── malignant (1).png
    │   ├── malignant (1)_mask.png
    │   └── ...
    └── normal/
        ├── normal (1).png
        ├── normal (1)_mask.png 
        └── ...
```

## Solution:


## U-net

base model: https://github.com/milesial/Pytorch-UNet <br>
torch hub model: https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/mateuszbuda_brain-segmentation-pytorch_unet.ipynb#scrollTo=a50761ab
Working Unet from PyTorch: https://drive.google.com/drive/folders/1EWM-WX8GLZYZvp0joQc5j-rzB18B3-wD?usp=sharing
