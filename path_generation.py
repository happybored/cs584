import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset,WeightedRandomSampler
import torch.optim as optim
import numpy as np
import sys
from lut import Sorted_LUT
from so6 import SO6,SO6_basis
from scipy.linalg import logm
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class SO6Env:
    goal_state: SO6
    state: SO6

    def __init__(self,t_step = 5,initial = 'random'):

        self.actions = []
        self.histoty_path = []
        self.t_step = t_step
        self.initial = initial
        self.reset()
        
    def reset(self):
        self.state = SO6(initialize='identity')
        if self.initial == 'random':
            random_numbers = np.random.randint(0, 15, self.t_step)
        else:
            self.initial_circuits = read_circ_file(f'data/{self.t_step}.dat')
            index = random.randrange(0, len(self.initial_circuits)-1)
            random_numbers = self.initial_circuits[index]
        # print(random_numbers)
        for num in random_numbers:
            self.state.left_multiply_by_T(num)


        self.state.canonical_form()
        return self.state.get_arr_after_perm().flatten()

    
    def is_done(self):
        lde = self.state.get_lde()
        if lde <=1:
            return True
        else:
            return False


    def step(self, action_index):
        assert self.state is not None, "Environment must be initialized before taking a step."

        self.state.left_multiply_by_T(action_index)

        done = self.is_done()
        reward = -1.0
        
        if done:
            reward += 1000.0
            print('done')
        self.state.canonical_form()
        return self.state.get_arr_after_perm().flatten(), reward, done


def read_circ_file(file_path):
    data_lists = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = list(map(int, line.strip().split()))
            data_lists.append(numbers)
    return data_lists

    
class MLP(nn.Module):
    def __init__(self, in_dim=36*3, out_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        )
        self.fc2 = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        )
        self.fc4 = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        )
        self.fc3 = nn.Linear(512, out_dim )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # training()
    # sys.exit(0)

    model = MLP(in_dim=108,out_dim=15)
    model.load_state_dict(torch.load("model/step_net_afterperm_108.pth"))

    ## Test the model by generating a path
    env = SO6Env(t_step= 10,initial='random')  # LUT[:10]
    # env = SO6Env(t_step= 10,initial= None)   # Random 10

    #test  phase 
    num_test_episodes = 100  
    test_max_step = 100
    
    test_episode_durations = []
    test_rewards = []
    successful_count = 0
    for episode in range(num_test_episodes):
        test_matrix = env.reset()
        
        done = False
        steps = 0
        total_reward = 0
        
        state = torch.tensor(test_matrix, dtype=torch.float32, device=device)
        
        while not done and steps < test_max_step:  # Limit steps to avoid infinite loops

            test_matrix = torch.tensor(test_matrix, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_index = model(test_matrix).argmax().item()  # Select best action
            
            test_matrix, reward, done = env.step(action_index)  # Apply action
            if done:
                successful_count = successful_count +1
            total_reward += reward
            steps += 1
        
        test_episode_durations.append(steps)
        test_rewards.append(total_reward)
    
    print('return rate:', successful_count/num_test_episodes)
    print('average length:', np.mean(test_episode_durations))