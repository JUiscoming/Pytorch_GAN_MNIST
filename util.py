# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:56:46 2020

@author: MSPL
"""
import torch

def CUDA(gpu_idx = 0):   
    if torch.cuda.is_available():
        if torch.cuda.device_count():
            print('CUDA is available, GPUs are also available.')
            torch.cuda.device(gpu_idx)
            print(torch.cuda.get_device_name(gpu_idx), 'is selected.')
            return str(gpu_idx)
        else:
            print('CUDA is installed, but GPUs are not available.')
            return -1
    else:
        return -1