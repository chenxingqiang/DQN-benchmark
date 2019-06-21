# [PYTORCH] Deep Q-learning Pytorch with flappy bird

## Introduction

Here is my python source code for training an agent to play flappy bird. It could be seen as a very basic example of Reinforcement Learning's application.
<p align="center">
  <img src="show/flappybird.gif" width=600><br/>
  <i>demo</i>
</p>  

## How to use my code

With my code, you can:
* **Train your model from scratch** by running **bash train.sh**
* **Test your trained model** by running **bash test.sh**

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
 Methods | Game | Mean score(second)
:--:|:--:|:--:
 DQN   | flappy bird | 61.2s
 Double DQN         | flappy bird | >160s
 Duel DQN      | flappy bird  | > 150S
 Duel Double DQN| flappy bird | >300s


## Requirements

* **python 3.6**
* **pygame**
* **cv2**
* **pytorch-1.0** 
* **numpy**

## Reference
- [DQN2013](https://arxiv.org/pdf/1312.5602.pdf)
- [DQN2015](https://www.nature.com/articles/nature14236.pdf)
- [Double DQN](https://arxiv.org/pdf/1509.06461v3.pdf)
- [Duel DQN](https://arxiv.org/pdf/1511.06581.pdf)


