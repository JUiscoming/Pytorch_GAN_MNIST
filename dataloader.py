# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:28:46 2020

@author: MSPL
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_MNIST(batch_size):
    transform = transforms.Compose([ #transforms.Compose is container of transforms.
            transforms.ToTensor(),  #torchvision.transforms.ToTensor: Convert a PIL Image of numpy.ndarray to tensor (H,W,C)->(C,H,W) in the range [0.0, 1.0].
            transforms.Normalize(mean = (0.5, ), std = (0.5, ))
            ])
    MNIST = datasets.MNIST(root = 'dataset', transform = transform, download = True)
    
    return DataLoader(MNIST, batch_size = batch_size, shuffle = True)