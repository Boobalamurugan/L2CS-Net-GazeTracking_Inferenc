import sys
import os
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import scipy.io as sio
import cv2
from torchvision import transforms

from .model import L2CS  # Importing L2CS gaze estimation model

# -------------------------------
# 📌 Image Preprocessing Pipeline
# -------------------------------
transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet standard deviation
    )
])



def prep_input_numpy(img: np.ndarray, device: str):
    """
    Prepare a Numpy image array as input to L2CS-Net.

    Args:
        img (np.ndarray): Input image or batch of images.
        device (str): Device ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    if len(img.shape) == 4:  # Batch of images
        img = torch.stack([transformations(im) for im in img])
    else:
        img = transformations(img)

    img = img.to(device)

    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    return img

def getArch(arch, bins):
    """
    Get the appropriate ResNet architecture for L2CS-Net.

    Args:
        arch (str): ResNet architecture ('ResNet18', 'ResNet34', etc.).
        bins (int): Number of gaze bins.

    Returns:
        L2CS: The selected model.
    """
    resnet_types = {
        'ResNet18': [2, 2, 2, 2],
        'ResNet34': [3, 4, 6, 3],
        'ResNet50': [3, 4, 6, 3],
        'ResNet101': [3, 4, 23, 3],
        'ResNet152': [3, 8, 36, 3],
    }

    if arch not in resnet_types:
        print(f'Invalid architecture "{arch}"! Defaulting to ResNet50.')
        arch = 'ResNet50'

    layers = resnet_types[arch]
    block_type = torchvision.models.resnet.Bottleneck if arch in ['ResNet50', 'ResNet101', 'ResNet152'] else torchvision.models.resnet.BasicBlock
    
    return L2CS(block_type, layers, bins)
