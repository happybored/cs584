import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset,WeightedRandomSampler
import torch.optim as optim
import numpy as np
import sys
from lut import Sorted_LUT
from so6 import SO6,SO6_basis
from scipy.linalg import logm
import time
import copy
import pickle

def read_dat_file(file_path):
    data_lists = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            numbers = list(map(int, parts[0].strip().split()))
            x = numbers[-1]
            a = numbers
            data_lists.append((x,a))
    return data_lists

def read_circ_file(file_path):
    data_lists = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = list(map(int, line.strip().split()))
            x = numbers[-1]
            data_lists.append((x,numbers))
    return data_lists

       
def preprocessing(data_path,N =None):
    dataset  = read_dat_file(data_path)
    X = []
    y = []
    if N is not None:
        dataset = dataset[:N]
    for  step,circuit in dataset:
        mat = SO6(initialize='identity')
        for c in circuit:
            mat.left_multiply_by_T(c)
        mat.canonical_form()
        X_temp = mat.get_arr_after_perm()
        X.append(X_temp)
        y.append(step)
    X = np.array(X)
    y = np.array(y)
    print('done: data generation')
    return X,y


def generate_data():
    layers = 8
    lut = Sorted_LUT()
    
    start_time = time.time()
    lut.constructed_from_file('data',layers)
    diff_time = time.time() - start_time
    
    for index in range(lut.layers):
        print(f'At layer {index+1}, the size is {len(lut.lookup_table[index+1])}')
    print("lut constructed, cost time:",diff_time)
    
    X = []
    X2 = []
    y = []
    y_last =[]
    for i in range(2,layers+1):
        print(f"at the {i} layer")
        count = 0
        for so6 in lut.lookup_table[i]:
            count= count+1
            print(count)
            circuit_list = so6.circuit_list
            old_lde = so6.get_lde()
            feasable_list = list()
            for j in range(15):
                circuit_list_temp = circuit_list +[j]
                new_mat =  SO6(initialize='circuit',args=circuit_list_temp)
                new_lde =  new_mat.get_lde()
                if new_mat in lut.lookup_table[i-1] and new_lde<=old_lde:
                    feasable_list.append(j)
            arr = so6.get_arr_after_perm()
            arr2 = so6.get_arr()
            X.append(arr)
            X2.append(arr2)
            y.append(feasable_list)
            y_last.append(circuit_list[-1])
    
    
        with open('output/X_after_perm.pkl', 'wb') as f:
            pickle.dump(X, f)
        
        with open('output/X_before_perm.pkl', 'wb') as f:
            pickle.dump(X2, f)
        
        with open('output/y_feasable.pkl', 'wb') as f:
            pickle.dump(y, f)
        with open('output/y_last.pkl', 'wb') as f:
            pickle.dump(y_last, f)
    
generate_data()

with open('output/y_feasable.pkl', 'rb') as f:
    y_feasable = pickle.load(f)

with open('output/y_last.pkl', 'rb') as f:
    y_last = pickle.load(f)

with open('output/X_after_perm.pkl', 'rb') as f:
    X = pickle.load(f)

print('test:')

sum = 0
for lable, feasable_list in zip(y_last,y_feasable):
    if lable not in feasable_list:
        sum = sum+1
print(sum)

X = np.array(X)
X_exp = X[:,:,2]
X_lde = np.max(X_exp,axis=1)
X_lde_pattern = (X_exp == X_lde[:, np.newaxis]).astype(int)

with open('output/X_exp.pkl', 'wb') as f:
    pickle.dump(X_exp, f)
with open('output/X_lde.pkl', 'wb') as f:
    pickle.dump(X_lde, f)
with open('output/X_lde_pattern.pkl', 'wb') as f:
    pickle.dump(X_lde_pattern, f)




# Create empty (n, 15) matrix
one_hot = np.zeros((len(y_feasable), 15), dtype=np.float32)

# Set 1s at the specified indices
for i, indices in enumerate(y_feasable):
    one_hot[i, indices] = 1.0

with open('output/y_feasable_onehot.pkl', 'wb') as f:
    pickle.dump(one_hot, f)
