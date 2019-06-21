#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2019-06-21 14:31
# * Last modified : 2019-06-21 14:31
# * Filename      : DQN.py
# * Description   : 
# **********************************************************
import argparse
import os
import shutil
from random import random, randint, sample
import numpy as np
import torch
import torch.nn as nn  
from demo import img_net
import cv2
from utils import op_util
class DQN(object):
    def __init__(self,opts):
        super(DQN,self).__init__()
        self.eval_net = getattr(img_net,opts.model)()
        self.target_net = getattr(img_net,opts.model)()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.init_ep = opts.initial_epsilon
        self.final_epsilon = opts.final_epsilon
        self.epsion = self.init_ep
        self.num_iters = opts.num_iters
        self.memory = []
        self.memory_size = opts.replay_memory_size
        self.target_replace_step = opts.replace
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = opts.batch_size
        self.gamma = opts.gamma
        self.warm_up = opts.warm_up
        self.clip_grad = opts.clip_grad and opts.model is "DuelDQN"
        #optimizer 
        if opts.optimizer is "adam":
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr = opts.lr)
        else:
            self.optimizer = torch.optim.SGD(self.eval_net.parameters(),lr = opts.lr,weight_decay=1e-4)
        self.criterion = nn.MSELoss()
    def save_transaction(self,s,a,r,s_,terminal):
        if len(self.memory)>self.memory_size:
            del self.memory[0]
        self.memory.append([s,a,r,s_,terminal])
    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
    def choice_action(self,x): 
        if (self.epsion)>np.random.uniform():
            return np.random.choice([0,0,0,0,0,0,0,0,1,1],1)[0]
        else:
            with torch.no_grad():
                action_val = self.dqn(x)
                return  (torch.max(action_val,1)[1]).cpu().numpy()[0]
    def learn(self):
        batch = sample(self.memory,self.batch_size)
        state_batch,action,reward,next_state_batch,ter_batch = zip(*batch)
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(np.asarray([a for a in action])).long().view(-1,1)
        reward_batch = torch.from_numpy(np.asarray(reward,dtype=np.float32)).view(-1,1)
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))
        if torch.cuda.is_available:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        q_eval = self.eval_net(state_batch).gather(1,action_batch)
        with torch.no_grad():
            target_action_val = self.target_net(next_state_batch)
            target_action_val = target_action_val.max(1)[1]
        q_target =  tuple(reward if terminal else reward+self.gamma * action_val for reward,terminal,action_val in zip(reward_batch,ter_batch,target_action_val))
        q_target = torch.cat(q_target).view(-1,1)
        self.optimizer.zero_grad()
        loss = self.criterion(q_eval,q_target)
        loss.backward()
        if self.clip_grad is True:
            op_util.scale_grad(self.optimizer,'conv')
        self.optimizer.step()
        return loss
    def ep_update(self,iter):
        self.epsion = self.final_epsilon+(self.num_iters-iter)*(1.*self.init_ep-self.final_epsilon)/self.num_iters
    def cuda(self):
        self.target_net.cuda()
        self.eval_net.cuda()
    def inference(self,x):
        x = self.dqn(x)
        return x.max(1)[1]
