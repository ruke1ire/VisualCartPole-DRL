#!/usr/local/bin/python3

import torch
from model import *
from env import Environment
from itertools import count
import random

device = 'cpu'

# create an environment
env = Environment()
env.reset()

# Actor Critic
#policy_net = PolicyNetwork(h=40,  w =150)
#policy_net.load_state_dict(torch.load('models/AC-RM-2 Q_LR=0.001 P_LR=5e-06 BATCH_SIZE=128 GAMMA=0.99 MEMORY_SIZE=150000 BETA=0.05_policy_4030.pt', map_location = 'cpu'))

# Policy Gradient
policy_net = PG_Network(h=40, w=150)
policy_net.load_state_dict(torch.load('models/PG hidden=64 lr=1e-05 max_steps=1000 Adam fullwidth BETA=0.01.pt', map_location = 'cpu'))

policy_net = policy_net.eval()

print(policy_net)

for episode in count():

    env.reset()
    prev_screen = env.get_screen().to(device)
    
    reward_sum = 0

    for steps in count():
        screen = env.get_screen().to(device)
        
        x = torch.cat((prev_screen, screen), dim=1)
        
        action_prob = policy_net(x)
        #action = 0 if random.random() < action_prob[0][0] else 1
        action = 0 if 0.5 < action_prob[0][0] else 1

        state, reward, done, _ = env.step(action)

        reward_sum += reward
        
        prev_screen = screen
        
        if done:
            print(f"[EPISODE: {episode}] [STEP: {reward_sum}]")
            break

