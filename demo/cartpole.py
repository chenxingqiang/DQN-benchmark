import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS= env.action_space.n
N_STATES = env.observation_space.shape[0]
LR=0.01
BATCH_SIZE = 32
EPSION = 0.5
GAMMA = 0.9
TARGET_REPLACE_ITER =100
MEMORY_CAPACTIY = 2000

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(N_STATES,10)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(10,N_ACTIONS)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        action_value = self.out(x)
        return action_value
class DQN(object):
    '''
    remeber here we must two policy
    fixed target
    memory reuse
    '''
    def __init__(self):
       self.eval_net,self.target_net = Net(),Net()
       self.learning_step_cout = 0
       self.memory_counter = 0
       self.memory = np.zeros((MEMORY_CAPACTIY,N_STATES*2+2))
       self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)
       self.eval_func = nn.MSELoss()
       self.epsion = EPSION
    def choose_action(self,x):
        if np.random.uniform()<self.epsion:
            with torch.no_grad():
                x = torch.unsqueeze(torch.Tensor(x).float(),0)
                action_value = self.eval_net(x)
                action = torch.max(action_value,1)[1].numpy()[0]
        else:
            action = np.random.randint(0,N_ACTIONS)
        return action
    def store_transaction(self,s,a,r,s_):
        transaction = np.hstack((s,[a,r],s_))
        index = self.memory_counter % MEMORY_CAPACTIY
        self.memory[index,:] = transaction
        self.memory_counter +=1
    def learn(self):
        if self.learning_step_cout % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        sample_index = np.random.choice(MEMORY_CAPACTIY,BATCH_SIZE)
        b_memory = self.memory[sample_index,:]
        b_s = torch.Tensor(b_memory[:,:N_STATES]).float()
        b_a = torch.Tensor(b_memory[:,N_STATES:N_STATES+1]).long()
        b_r = torch.Tensor(b_memory[:,N_STATES+1:N_STATES+2]).float()
        b_s_ = torch.Tensor(b_memory[:,-N_STATES:]).float()
        q_eval =self.eval_net(b_s).gather(1,b_a)
        with torch.no_grad():
            q_next = self.target_net(b_s_)
        q_target = b_r+GAMMA*q_next.max(1)[0]
        loss = self.eval_func(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_step_cout+=1
    def inference(self,x):
        with torch.no_grad():
            x = torch.unsqueeze(torch.Tensor(x),0).float() 
            eval_val = self.eval_net(x)
            return eval_val.max(1)[1].squeeze(0).numpy()

def  train():
    dqn = DQN()
    for i_episode in range(20000):
        s = env.reset()
        dqn.epsion = 0.5+0.5*i_episode/20000
        print("eposide:{}".format(i_episode))
        while True:
            a = dqn.choose_action(s)
            s_,r,done,info = env.step(a)
            x,x_dot,theta,theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5 
            r = r1 + r2
            dqn.store_transaction(s,a,r,s_)
            if dqn.memory_counter>MEMORY_CAPACTIY:
                dqn.learn()
            if done:
                break
            s = s_
    torch.save(dqn.eval_net.state_dict(),"./final.pth")
def test():
    dqn =DQN()
    dqn.eval_net.load_state_dict(torch.load("./final.pth"))
    while True:
        s = env.reset()
        while True:
            env.render()
            a = dqn.inference(s)
            s_,r,done,info = env.step(a)
            if done:
                print("game over")
                break
            else:
                s =s_
if __name__ =="__main__":
    #train()
    test()


