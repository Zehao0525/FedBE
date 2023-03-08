import torch
import torch.nn as nn
from torchvision import datasets, transforms

class GaussNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def apply_data_shift(trans,rotation = 0., translation = 0., noise = [0.,0.],flip = 0.):
    if rotation !=0:
        trans = transforms.Compose([trans,transforms.RandomRotation(rotation)])
    if noise != [0.,0.]:
        trans = transforms.Compose([
            trans,
            GaussNoise(mean = noise[0],std = noise[1]),
        ])
    if flip > 0.5 :
        trans = transforms.Compose([trans,transforms.RandomHorizontalFlip(flip)])
    return trans




def load_data(dataset, rotation = 0., translation = 0., noise = [0.,0.], flip = 0.,trian = False):
    if type(noise) == float:
        noise = [0.,noise]
    if dataset == 'cifar10':
        trans = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trans = apply_data_shift(trans,rotation, translation, noise, flip)
        dataset_test = datasets.CIFAR10('../data/cifar', train=trian, download=True, transform=trans)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trans = apply_data_shift(trans,rotation, translation, noise, flip)
        dataset_test = datasets.MNIST('../data/mnist/', train=trian, download=True, transform=trans)
        classes = ('0','1','2','3','4','5','6','7','8','9')
    elif dataset == 'svhn':
        trans = transforms.Compose([
          transforms.Grayscale(num_output_channels=1),
          transforms.Resize((28,28)),
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
        ])
        trans = apply_data_shift(trans,rotation, translation, noise, flip)
        dataset_test = datasets.SVHN('../data/svhn/', train=trian, download=True, transform=trans)
        classes = ('0','1','2','3','4','5','6','7','8','9')
    else:
        exit('Error: unrecognized dataset')

    
    return dataset_test,classes