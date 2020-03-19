# Pytorch_GAN_MNIST

>GAN Epoch 100: random 100 samples

![fake_epoch_100](https://user-images.githubusercontent.com/33240322/77092658-5471ed80-6a4d-11ea-99b7-c3bcac55adcf.png)

>CGAN Epoch 100: random 100 samples

![fake_epoch_100](https://user-images.githubusercontent.com/33240322/77092697-5cca2880-6a4d-11ea-9159-ce6e1280e5ee.png)

>CGAN Epoch 100 and 10 samples with condition [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

![Figure_1](https://user-images.githubusercontent.com/33240322/77092718-62277300-6a4d-11ea-8ce6-18d05b1dd07b.png)

## Usage

execute the 'main.py' using IPython.
- model: 'GAN' or 'CGAN'
- train: True or False(test)
- epoch uses when loading your traind model. ex) if you want to load 'GAN_epoch_100.pkl', epoch = 100.

## Paper

Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014. [Link](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014). [Link](https://arxiv.org/pdf/1411.1784.pdf)
