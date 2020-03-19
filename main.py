# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:42:07 2020

@author: MSPL
"""

import torch
from util import CUDA
from GAN import GAN
from CGAN import CGAN


def main():
    model = 'CGAN'
    train = False
    epoch = 100
    
    if model == 'GAN':
        kwargs = {'epochs': 100,
                'batch_size': 100,
                'input_dim': 100,
                'data_dim': 28*28,
                'gpu_number': CUDA(gpu_idx = 0),
                'lrG': 2e-4,
                'lrD': 2e-4,
                'beta1': 0.5,
                'beta2': 0.999,
                'saving_epoch_interval': 2
                }      
        gan = GAN(**kwargs)        
    elif model == 'CGAN':
        kwargs = {'epochs': 100,
                'batch_size': 100,
                'input_dim': 100,
                'data_dim': 28*28,
                'condition_dim': 10,
                'gpu_number': CUDA(gpu_idx = 0),
                'lrG': 2e-4,
                'lrD': 2e-4,
                'beta1': 0.5,
                'beta2': 0.999,
                'saving_epoch_interval': 2
                }      
        gan = CGAN(**kwargs)
    if train:
        gan.load(epoch = epoch)
        gan.train()    
    else:
        gan.load(epoch = epoch)
        gan.test()
        

if __name__ == '__main__':
    main()
