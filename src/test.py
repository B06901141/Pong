from environment import Pong

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import DNN

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("model_name")
args = parser.parse_args()

env = Pong()
env.reset()
init, _, _ = env.step(0)

s = init
pre_s = init

action_space = [0,2,3] #[stay, up ,down]
action_num = len(action_space)

model = DNN()
model.load_state_dict(torch.load(args.model_name))
model.cuda()

while True:
    done = False
    env.reset()
    env.render()
    while not done:
        time.sleep(0.01)
        env.render()
        output = model(torch.Tensor(s-pre_s).cuda())
        index = output.argmax().item()
        action = action_space[index]
        pre_s = s
        s, r, done = env.step(action)

