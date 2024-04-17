import torch
from torch import nn, optim, functional, utils
import torchvision
from torchvision import datasets, utils

import time, os


class AutoEncoderLayer(nn.Module):
    """
    fully-connected linear layers for stacked autoencoders.
    This module can automatically be trained when training each layer is enabled
    Yes, this is much like the simplest auto-encoder
    """

    def __init__(self, input_dim=None, output_dim=None, SelfTraining=False, sparsity_ratio=None):
        super(AutoEncoderLayer, self).__init__()
        # if input_dim is None or output_dim is None:
        #     raise ValueError
        self.in_features = input_dim
        self.out_features = output_dim
        self.is_training_self = SelfTraining  # 指示是否进行逐层预训练,还是训练整个网络
        self.sparsity_ratio = sparsity_ratio
        # 编码器
        self.encoder = nn.Linear(self.in_features, self.out_features, bias=True)  # 统一使用Sigmoid激活
        # 解码器
        self.decoder = nn.Linear(self.out_features, self.in_features, bias=True)  # 此处decoder不使用encoder的转置, 并使用Sigmoid进行激活.

    def forward(self, x):         # 如果是自训练，输出译码器结果；若否，输出编码器结果（代表输出隐藏层）
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if self.is_training_self:
            return decoded, encoded
            #return self.decoder(encoded)
        else:
            return encoded

    def sparse_loss(self,encoded):
        #sparsity = torch.mean(encoded,dim=0)
        sparsity = torch.sigmoid(encoded)
        kl_div = self.sparsity_ratio * torch.log(self.sparsity_ratio/sparsity) + \
                 (1 - self.sparsity_ratio) * torch.log( (1-self.sparsity_ratio) / (1 - sparsity) )
        return kl_div.sum()


    def lock_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def acquire_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    @property
    def is_training_layer(self):
        return self.is_training_self

    @is_training_layer.setter
    def is_training_layer(self, other: bool):
        self.is_training_self = other


class StackedAutoEncoder(nn.Module):
    """
    Construct the whole network with layers_list
    > 栈式自编码器的架构一般是关于中间隐层对称的
    """

    def __init__(self, layers_list=None):
        super(StackedAutoEncoder, self).__init__()
        self.layers_list = layers_list
        self.initialize()
        self.encoder_1 = self.layers_list[0]
        self.encoder_2 = self.layers_list[1]
        self.encoder_3 = self.layers_list[2]
        self.encoder_4 = self.layers_list[3]

    def initialize(self):
        for layer in self.layers_list:
            # assert isinstance(layer, AutoEncoderLayer)
            layer.is_training_layer = False
            # for param in layer.parameters():
            #     param.requires_grad = True

    def forward(self, x):
        out = x
        # for layer in self.layers_list:
        #     out = layer(out)
        out = self.encoder_1(out)
        out = self.encoder_2(out)
        out = self.encoder_3(out)
        out = self.encoder_4(out)
        return out


class myAutoEncoderLayer(nn.Module):
    """
    fully-connected linear layers for stacked autoencoders.
    This module can automatically be trained when training each layer is enabled
    Yes, this is much like the simplest auto-encoder
    """

    def __init__(self, input_dim=None,output_dim=None, SelfTraining=False):
        super(myAutoEncoderLayer, self).__init__()
        # if input_dim is None or output_dim is None:
        #     raise ValueError
        self.in_features = input_dim
        self.hidden_features = output_dim
        self.is_training_self = SelfTraining  # 指示是否进行逐层预训练,还是训练整个网络
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features, bias=True) # 统一使用Sigmoid激活
        )
        self.decoder = nn.Sequential(  # 此处decoder不使用encoder的转置, 并使用Sigmoid进行激活.
            nn.Linear(self.hidden_features, self.in_features, bias=True)
        )

    def forward(self, x):
        out = self.encoder(x)
        if self.is_training_self:
            return self.decoder(out)
        else:
            return out

    def lock_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def acquire_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    @property
    def is_training_layer(self):
        return self.is_training_self

    @is_training_layer.setter
    def is_training_layer(self, other: bool):
        self.is_training_self = other