#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2019-06-21 14:29
# * Last modified : 2019-06-21 14:29
# * Filename      : train.py
# * Description   : 
# **********************************************************
import argparse
import os
import shutil
from random import random, randint, sample
import os
import numpy as np
import torch
import torch.nn as nn  
from board import Visualizer
from demo import img_net
from database.flappy_bird import FlappyBird
import cv2
import model
from model import DDQN,DQN
def pre_processing(image, width, height):
    img = cv2.resize(image,(width,height))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = img[None, :, :].astype(np.float32)
    return torch.from_numpy(img)

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=200000)
    parser.add_argument("--replace", type=int, default=200)
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--vis", type=bool, default=True)
    parser.add_argument("--model", type=str, default="DuelDQN")
    parser.add_argument("--train_model", type=str, default="DDQN")
    parser.add_argument("--warm_up", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="weight")
    parser.add_argument("--clip_grad", type=bool, default=True)
    parser.add_argument("--Fire", type=str,choices=["train","test"],default="train")


    args = parser.parse_args()
    return args
def tostate(img,game,opts):
    image = pre_processing(img[:game.screen_width,:int(game.base_y)],opts.image_size,opts.image_size)
    return image
def inference(opts):
    agent = getattr(getattr(model,opts.train_model),opts.train_model)(opts)
    game = FlappyBird()
    img,_,_ = game.next_frame(0)
    image = pre_processing(img[:game.screen_width,:int(game.base_y)],opts.image_size,opts.image_size)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    if torch.cuda.is_available():
        state = state.cuda()
        agent.cuda()
    terminal = False
    load_dir = os.path.join(opts.save_dir,"{}_{}.pth".format(opts.train_model,opts.model))
    agent.eval_net.load_state_dict(torch.load(load_dir))
    while True:
            a  = agent.inference(state)[0]
            next_image,r,terminal = game.next_frame(a)
            next_image = tostate(next_image,game,opts)
            if torch.cuda.is_available():
                next_image = next_image.cuda()
            next_state = torch.cat((state[0,1:,:,:],next_image))[None,:,:,:]
            state = next_state

def train(opts):
    agent = getattr(getattr(model,opts.train_model),opts.train_model)(opts)
    game = FlappyBird()
    img,_,_ = game.next_frame(0)
    image = pre_processing(img[:game.screen_width,:int(game.base_y)],opts.image_size,opts.image_size)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    if torch.cuda.is_available():
        state = state.cuda()
    print("agent trainnig")
    print("iter:{}".format(agent.num_iters))
    print("replace_time:{}".format(agent.target_replace_step))
    print("memory_size:{}".format(agent.memory_size))
    agent.epsion = 1.
    if torch.cuda.is_available():
        agent.cuda()
    for ite in range(1,agent.warm_up+agent.num_iters+1):
        a = agent.choice_action(state)
        next_image,reward,terminal = game.next_frame(a)
        next_image = tostate(next_image,game,opts)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0,1:,:,:],next_image))[None,:,:,:]
        agent.save_transaction(state,a,reward,next_state,terminal)
        if ite<=agent.warm_up:
            print("warm_up,ite:{},epsion{},action:{},terminal:{}".format(ite,agent.epsion,a,terminal))
        else:
            agent.ep_update(ite-agent.warm_up)
            if (ite-agent.warm_up) % agent.target_replace_step == 0:
                agent.update_target()
                print("update target model")
            loss = agent.learn()
            print("iteration:{},epsion:{},action:{},reward:{},terminal:{},loss:{}".format(ite,agent.epsion,a,reward,terminal,loss))
        state = next_state
    torch.save(agent.eval_net.state_dict(),os.path.join(opts.save_dir,"{}_{}.pth".format(opts.train_model,opts.model)))
if __name__ == "__main__":
    opts = get_args()
    if opts.Fire == "train":
        train(opts)
    else:
        inference(opts)

