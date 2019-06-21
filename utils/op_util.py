#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2019-06-21 14:29
# * Last modified : 2019-06-21 14:29
# * Filename      : op_util.py
# * Description   : 
# **********************************************************
import torch.nn as nn
import torch
def group_weight(weight_group, module, norm_layer, lr,name = None):
    group_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_decay.append(m.weight)
            if m.bias is not None:
                group_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay)
    weight_group.append(dict(params=group_decay, lr=lr,name=name))
    return weight_group
def clip_gradient(optimizer,grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clip_grad_norm_(-grad_clip,grad_clip)
def clip_norm_grad(model,grad_clip):
    return nn.utils.clip_grad_norm_(model.parameters(),grad_clip,2)
def scale_grad(optimizer,name,val):
    for group in optimizer.param_groups:
        if group["name"] is name:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(val)
