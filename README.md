# [PYTORCH] Deep Q-learning Pytorch Realize

## Introduction

Here is my python source code for training an agent to play flappy bird. It could be seen as a very basic example of Reinforcement Learning's application.
<p align="center">
  <img src="show/flappybird.gif" width=600><br/>
  <i>demo</i>
</p>

## How to use my code

With my code, you can:
* **Train your model from scratch** by running **bash train.sh**
* **Test your trained model** by running **bash test.py**

## Trained models
### train with python
```
python ./tool/train.py --model DuelDQN --train_model DDQN --clip_grad True --Fire train
#python ./tool/train.py --model DQN --train_model DDQN --clip_grad True --Fire train
```
--model means network structure  
--train_model means using DQN train-policy or double DQN train-policy   
### train with shell
You could find DDQN model at **model/DDQN**  
You could find DQN  model at **model/DQN**

## NetWork
You could find onestream at **demo/img_net--->DQN**  
You could find duel-stream at **demo/img_net--->DuelDQN**  


## Result

## Requirements

* **python 3.6**
* **pygame**
* **cv2**
* **pytorch-1.0** 
* **numpy**
