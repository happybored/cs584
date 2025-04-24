from so6 import SO6
from sortedcontainers import SortedList
import sys
import numpy as np
import copy
import math
import time

np.set_printoptions(linewidth=100)

def read_dat_file(file_path):
    data_lists = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = list(map(int, line.split()))  # Convert space-separated values to a list of integers
            data_lists.append(numbers)
    return data_lists


class Sorted_LUT:
    def __init__(self, root = None):
        self.layers = 0
        self.lookup_table = None
        if root == None:
            self.root = SO6(initialize='identity')
        else:
            self.root = root

    def insert(self, layer, value, debug = False):
        if value not in self.lookup_table[layer]:
            self.lookup_table[layer].add(value)
        else:
            if debug:
                other_index = self.lookup_table[layer].index(value)
                other = self.lookup_table[layer][other_index]
                print(f'insert the same elements at layer {layer+1}, where circuit is {value.circuit_list}, and the existing circuit is {other.circuit_list} ')

    def contain(self, value):
        for  layer in range(self.layers+1):
            if value in self.lookup_table[layer]:
                return layer
        return -1

    
    def constructed_from_file(self,fold_path,num_layers):
        self.layers = num_layers
        self.lookup_table = [SortedList() for _ in range(num_layers +1)]
        self.insert(0,self.root)

        for index in range(1,self.layers+1):
            file_path = f"{fold_path}/{index}.dat"  # Change this to your actual file path
            circuits = read_dat_file(file_path)
            for circuit in circuits:
                mat =  SO6(initialize='circuit',args=circuit)
                self.insert(index,mat)
        # print('constructed_from_file done')


    def initilize(self):
        self.lookup_table = [SortedList()]
        self.insert(0,self.root)

        self.layers = 0

    def get_next_T_count(self,inverse = False):
        self.lookup_table.append(SortedList())

        cur_layer = self.layers
        for mat in self.lookup_table[cur_layer]:
            # if mat.last_T == -1:
            for i in range(15):
                if inverse == False:
                    to_insert = mat.left_multiply_by_T(i,inplace = False)
                else:
                    to_insert = mat.left_multiply_by_inverse_T(i,inplace = False)

                if to_insert not in self.lookup_table[cur_layer-1]:
                    self.insert(cur_layer+1, to_insert)
        self.layers = self.layers +1

    def constructed_from_recursive(self, number_layers,inverse= False):
        #initilize
        self.initilize()
        for _ in range(number_layers):
            self.get_next_T_count(inverse = inverse)
        # print('constructed_from_recursive done')


    def to_dataset(self):
        X = []
        y = []
        for layer in range(self.layers):
            for so6 in self.lookup_table[layer]:
                X_temp = so6.get_arr()
                y_temp = layer 
                X.append(X_temp)
                y.append(y_temp)
        return np.array(X),np.array(y)
    

def meet_in_the_middle(mat:SO6,right_lut:Sorted_LUT):
    ret = right_lut.contain(mat)
    if ret>0:
        return ret

    left_lut = Sorted_LUT(root=mat)
    left_lut.constructed_from_recursive(number_layers= math.floor(right_lut.layers /2))

    for index in range(right_lut.layers):
        current_layer = right_lut.layers + index + 1
        right_layer = right_lut.layers
        left_layer =  current_layer - right_layer
        
        #construct left LUT step by step
        while left_lut.layers < left_layer:
            left_lut.get_next_T_count()
        
        # meet in the middle
        for T1 in right_lut.lookup_table[right_layer]:
            for T2 in left_lut.lookup_table[left_layer]:
                if T1 ==T2:
                    return current_layer
    return -1



def meet_in_the_middle2(mat:SO6,right_lut:Sorted_LUT):
    ret = right_lut.contain(mat)
    if ret>0:
        return ret
    min_ret = 100
    for S in right_lut.lookup_table[right_lut.layers]:
        temp = mat.transpose()
        temp = temp.left_mul(S)
        ret = right_lut.contain(temp)
        if ret>0 and ret< min_ret:
            min_ret = ret

    if min_ret==100:
        return -1
    else:  
        return min_ret + right_lut.layers