#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2019-06-21 14:30
# * Last modified : 2019-06-21 14:30
# * Filename      : img_net.py
# * Description   : img_type CNN network 
# **********************************************************
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self,action=2):
        super(DQN,self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(4,32,kernel_size = 8,stride =4),nn.ReLU(inplace = True))
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,kernel_size = 4,stride =2),nn.ReLU(inplace = True))
        self.conv3 = nn.Sequential(nn.Conv2d(64,64,kernel_size = 3,stride = 1),nn.ReLU(inplace = True))

        self.fc1 = nn.Sequential(nn.Linear(7*7*64,512),nn.ReLU(inplace = True))
        self.fc2 = nn.Linear(512,action)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.uniform(m.weight,-0.01,0.01)
                nn.init.constant_(m.bias,0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
class DuelDQN(nn.Module):
    def __init__(self,action = 2):
        super(DuelDQN,self).__init__()
        conv_layer = []
        conv_layer.append(nn.Sequential(nn.Conv2d(4,32,kernel_size = 8,stride =4),nn.ReLU(inplace = True)))
        conv_layer.append(nn.Sequential(nn.Conv2d(32,64,kernel_size = 4,stride =2),nn.ReLU(inplace = True)))
        conv_layer.append(nn.Sequential(nn.Conv2d(64,64,kernel_size = 3,stride = 1),nn.ReLU(inplace = True)))
        self.backbone = nn.Sequential(*conv_layer)
        self.business_layer = []
        self.v_fc1 = nn.Sequential(nn.Linear(7*7*64,512),nn.ReLU(inplace = True))
        self.adv_fc1 = nn.Sequential(nn.Linear(7*7*64,512),nn.ReLU(inplace = True))
        self.v_logits = nn.Linear(512,1)
        self.adv_logits = nn.Linear(512,action)
        self.business_layer.append(self.v_fc1)
        self.business_layer.append(self.adv_fc1)
        self.business_layer.append(self.v_logits)
        self.business_layer.append(self.adv_logits)
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.uniform(m.weight,-0.01,0.01)
                nn.init.constant_(m.bias,0)
    def forward(self,x):
        x = self.backbone(x)
        x = x.view(x.shape[0],-1)
        v1 = self.v_fc1(x)
        v = self.v_logits(v1)
        adv1 = self.adv_fc1(x)
        adv = self.adv_logits(adv1)
        return v+(adv-torch.mean(adv,dim=1,keepdim = True))
if __name__ == "__main__":
    dqn = DuelDQN()
    x = torch.randn(1,4,84,84)
    print(dqn(x))

