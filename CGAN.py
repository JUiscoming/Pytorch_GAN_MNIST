# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 04:01:20 2020

@author: MSPL
"""
import time
import datetime
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import load_MNIST
from torchvision.utils import save_image

#    Linear(input_dim, output_dim): There exist interaction expression from input to output.
#                                    each layer equals a sub-function
#                                    and combining all layers make a function.
#                                    Thinking Neural Networks as a function is better way of understanding ML.


#    1. When you train a model, think it first.
#        1) Loss function
#        2) Learning rate
#        3) Optimizer
#        4) Epoch: One Epoch is when an ENTIRE dataset is passed forward and backward through the Neural Networks only ONCE.
#           (Batch size: Total number of training examples present in a single batch)
#           (Iteration: The number of passes to complete one epoch, in each iteration training is operated; Iteration=ceil(#dataset / Batch size)
#
#    2. Basic code(each epoch, each iteration)
#        optimizer.zero_grad()
#        pred = model(x)
#        loss = criterion(pred, x_labels)
#        loss.backward()
#        optimizer.step()

# Generator
class Generator(nn.Module):

    def __init__(self, input_dim, data_dim, condition_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim  #dim of input layer
        self.data_dim = data_dim    #dim of data
        self.condition_dim = condition_dim

        self.fc = nn.Sequential(  #Container of nn.Module; sequentially operate nn.Module.
                # input -> layer1
                nn.Linear(self.input_dim + self.condition_dim, 256),
                nn.LeakyReLU(negative_slope = 0.2),
#                # layer1 -> layer2
                nn.Linear(256, 256),
                nn.LeakyReLU(negative_slope = 0.2),
                # layer2 -> output
                nn.Linear(256, self.data_dim),
                nn.Tanh()
                )

    def forward(self, input, condition):
        x = torch.cat([input, condition], 1)
        x = self.fc(x)

        return x

# Discriminator
class Discriminator(nn.Module):

    def __init__(self, data_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.data_dim = data_dim
        self.condition_dim = condition_dim

        self.fc = nn.Sequential(
                # input -> layer1
                nn.Linear(self.data_dim + self.condition_dim, 256),
                nn.LeakyReLU(negative_slope = 0.2),
#                # layer1 -> layer2
                nn.Linear(256, 256),
                nn.LeakyReLU(negative_slope = 0.2),
                # layer2 -> output
                nn.Linear(256, 1),
                nn.Sigmoid()
                )

    def forward(self, input, condition):
        x = torch.cat([input, condition], 1)
        x = self.fc(x)

        return x

class CGAN(object):
    def __init__(self, **kwargs):
        # parameters
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']
        self.data_dim = kwargs['data_dim']
        self.input_dim = kwargs['input_dim']
        self.condition_dim = kwargs['condition_dim']
        
        # other settings
        self.device = torch.device('cuda:'+kwargs['gpu_number'] if kwargs['gpu_number'] != -1 else 'cpu')
        self.loaded_epoch = 0
        self.model_name = 'CGAN'
        self.save_dir = ['models', 'CGAN_MNIST']
        self.sample_dir = os.path.join(*['samples', 'CGAN_MNIST'])
        self.saving_epoch_interval = kwargs['saving_epoch_interval']
        self.data_loader = load_MNIST(batch_size = self.batch_size)


        # Networks, Optimizers and Loss
        self.G = Generator(data_dim = self.data_dim, input_dim = self.input_dim, condition_dim = self.condition_dim).to(device = self.device)
        self.D = Discriminator(data_dim = self.data_dim, condition_dim = self.condition_dim).to(device = self.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr = kwargs['lrG'], betas = (kwargs['beta1'], kwargs['beta2']))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr = kwargs['lrD'], betas = (kwargs['beta1'], kwargs['beta2']))

        self.BCE_loss = nn.BCELoss().to(device=self.device)
        
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []

        self.real_labels = torch.ones(self.batch_size, 1).to(device=self.device)
        self.fake_labels = torch.zeros(self.batch_size, 1).to(device=self.device)

        #Train

        print('Training Starts.')

        for epoch in range(self.loaded_epoch, self.epochs):
            epoch_start_time = time.time()
            print('Epoch [%4d/%4d]:' % ((epoch + 1), self.epochs))
            for iter, (x, y) in enumerate(self.data_loader):   #_ means discard something; y is discarded.
                
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                x = x.view(-1, self.data_dim).to(device=self.device) #(-1, 28*28) flatten
                #z = (torch.randn((self.batch_size, self.input_dim)).to(device=self.device)-0.5)*2
                z = torch.randn((self.batch_size, self.input_dim)).to(device=self.device)
                # y.shape = [512]
                y = F.one_hot(y, num_classes = self.condition_dim).to(device=self.device)  # ex: [[0], [1], ...] = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ...]
                # y.shape = [512, 10]
                y = y.float()
                
                #**************************************#
                #              Training D              #
                #**************************************#

                # Compute D loss
                outputs = self.D(x, y)
                D_real_loss = self.BCE_loss(outputs, self.real_labels)
                real_score = outputs
                outputs = self.D(self.G(z, y), y)
                D_fake_loss = self.BCE_loss(outputs, self.fake_labels)
                D_loss = D_real_loss + D_fake_loss  # Loss of Discriminator
                fake_score = outputs
                
                # Backpropagate and Optimize
                self.train_hist['D_loss'].append(D_loss.item()) # tensor.item: returns the value of this tensor as a python data.
                self.D_optimizer.zero_grad()
                D_loss.backward()   # Calculate gradients.
                self.D_optimizer.step() # Performs a single optimization step.
                #**************************************#
                #              Training G              #
                #**************************************#
                # Compute G loss
                fake_images = self.G(z, y)
                outputs = self.D(fake_images, y)
                G_loss = self.BCE_loss(outputs, self.real_labels)
                
                # Backpropagate and Optimize
                self.train_hist['G_loss'].append(G_loss.item())
                self.G_optimizer.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 200) == 0:
                    print("  Iter [{}/{}]: D_loss: {:.4f}, G_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}"
                          .format(iter + 1, self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item(), real_score.mean().item(), fake_score.mean().item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            avg_epoch_time = int(np.mean(self.train_hist['per_epoch_time']))
            print("  Avg. 1 epoch time: [%s] / Est. remaining time: [%s]" % (str(datetime.timedelta(seconds = avg_epoch_time)),
                                                                                        str(datetime.timedelta(seconds = (self.epochs - epoch - 1)*avg_epoch_time))))
            if (epoch + 1) % self.saving_epoch_interval == 0:
                self.save(epoch)

            # Save sampled images
            fake_images = (fake_images + 1) / 2
            fake_images = fake_images.reshape(self.G(z, y).size(0), 1, 28, 28)
            
            save_image(fake_images, os.path.join(self.sample_dir, 'fake_epoch_{}.png'.format(epoch+1)))


        print("Training Finish.")


    def save(self, epoch):
        save_dir = os.path.join(*self.save_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save({
                'epoch': epoch,
                'G_state_dict': self.G.state_dict(),
                'D_state_dict': self.D.state_dict(),
                'G_optimizer_state_dict': self.G_optimizer.state_dict(),
                'D_optimizer_state_dict': self.D_optimizer.state_dict()
                }, os.path.join(save_dir, self.model_name + '_epoch_' + str(epoch + 1) + '.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
        print('  epoch:', epoch + 1, ', model saved!')

    def load(self, epoch):
        save_dir = os.path.join(*self.save_dir, self.model_name)

        if os.path.isfile(save_dir + '_epoch_' + str(epoch) + '.pkl'):
            checkpoint = torch.load(save_dir + '_epoch_' + str(epoch) + '.pkl')
            self.G.load_state_dict(checkpoint['G_state_dict'])
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
            self.loaded_epoch = checkpoint['epoch'] + 1 # loaded_epoch + 1 is start point.
            print('loading checkpoint successes!')
        else:
            print('no checkpoint found at %s' % save_dir)
            return

    def test(self, sample_num = 10):
        self.G.eval()

        z = torch.randn((sample_num, self.input_dim)).to(device=self.device)
        y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to(device = self.device)
        y = F.one_hot(y, num_classes = self.condition_dim).to(device=self.device)
        y = y.float()
        
        fake_image = self.G(z, y)

        if self.device != 'cpu':
            fake_image = fake_image.cpu().data.numpy()    #G_z.shape=[10, 100]
        else:
            fake_image = fake_image.data.numpy()

        fake_image = (fake_image + 1) / 2
        fake_image=fake_image.reshape((sample_num, 28, 28))

        rows = 3
        cols = 4

        for i in range(sample_num):
            plt.subplot(rows, cols, i+1)
            plt.imshow(fake_image[i], cmap='gray')
        plt.show()
