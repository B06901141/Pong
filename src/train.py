import numpy as np
import os

from environment import Pong

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import DNN

model_index = 1400
roll_off = 0.995
episode_per_update = 15
max_update = 2000

action_space = [0,2,3] #[stay, up ,down]
action_num = len(action_space)

env = Pong()
init, _, _ = env.step(0)

model = DNN()
if model_index == 0:
    os.system("rm -rf ../model")
    os.makedirs("../model",exist_ok=True)
    os.makedirs("../log",exist_ok=True)
    torch.save(model.state_dict(), "../model/init.pt")
    with open("../log/history.csv","w") as f:
        f.write("Update,Episode,Reward\n")
else:
    model.load_state_dict(torch.load("../model/update%d.pt"%model_index))

model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

log = ""
for update in range(model_index,max_update):
    log_prob = []
    reward = []
    total_reward = 0
    print("Collecting state..")
    for episode in range(1,episode_per_update + 1):
        done = False
        s = init
        pre_s = init
        env.reset()
        game_reward = 0
        while not done:
            output = model(torch.Tensor(s-pre_s).cuda())
            probability = F.softmax(output, dim = 1)
            
            C = torch.distributions.categorical.Categorical(probability)
            step = C.sample()
            action = action_space[step.item()]
            
            pre_s = s
            s, r, done = env.step(action)
            game_reward += r
            log_prob.append(C.log_prob(step))
            reward.append(r)

            if r != 0:
                s = init
                pre_s = init
        print("Update %d: Episode %d/%d, Reward = %d"%(update+1, episode, episode_per_update, game_reward))
        log += "%d,%d,%d\n"%(update+1, episode, game_reward)
        total_reward += game_reward
    
    reward = np.array(reward)
    num_data = reward.shape[0]
    print("Start training with %d data..."%num_data,end = '')
    total_reward /= episode_per_update
    
    r = 0
    for i in range(num_data-1,-1,-1):
        if reward[i] != 0:
            r = reward[i]
        else:
            r *= roll_off
        reward[i] = r
    reward = (reward - reward.mean())/(reward.std() + 1e-7)
        

    reward = Variable(torch.from_numpy(reward).type(torch.FloatTensor)).cuda()
    log_prob = - torch.cat(log_prob)

    loss = torch.sum(log_prob*reward)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print("Done")
        
    if update % 50 == 49:
        torch.save(model.state_dict(), "../model/update%d.pt"%(update+1))
        with open("../log/history.csv","a") as f:
            f.write(log)
        log = ""
    print("Update %d: Average reward = %.2f\n"%(update+1, total_reward))

