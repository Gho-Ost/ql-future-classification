# ql-future-solution

Our solution for the QL future hackathon 2024 problem. We set out to create a high-performing solution for the classification of breast cancer based on images with the help of quantum convolutional neural networks. 
Furthermore, we set out to show the possibility of quantum supremacy for the task of classification of diseases from images. 

## Team: AI've Got Genes

## Problem description:
Breast cancer is one of the most common causes of death among women worldwide. Early detection helps in reducing the number of early deaths. The data reviews the medical images of breast cancer using ultrasound scans. The Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images. Breast ultrasound images can produce great results in the classification, detection, and segmentation of breast cancer when combined with machine learning.

## Data:
Image recognition dataset:
The data collected at baseline include breast ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018. The number of patients is 600 female patients. The dataset consists of 780 images with an average image size of 500*500 pixels, although the sizes vary among images. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes, which are normal, benign, and malignant.


## drive links with models and more data:
https://drive.google.com/drive/folders/1ljjVvm4S7X6dXdtbS4wCe1CKylzP5UDz?usp=sharing
https://drive.google.com/drive/folders/1EWM-WX8GLZYZvp0joQc5j-rzB18B3-wD?usp=sharing

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


## U-net reference links:
base model: https://github.com/milesial/Pytorch-UNet <br>
torch hub model: https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/mateuszbuda_brain-segmentation-pytorch_unet.ipynb#scrollTo=a50761ab <br>

## References:
ResNet: https://arxiv.org/abs/1512.03385 <br>
Unet: https://arxiv.org/abs/1505.04597 <br>
QNN: https://arxiv.org/abs/2205.08154 <br>
QCNN: https://arxiv.org/abs/1810.03787 <br>
QCNN for classification - tutorial: https://www.frontiersin.org/articles/10.3389/fphy.2022.1069985/full#B31 <br>
