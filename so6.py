import struct
import numpy as np
import math
from typing import Any, Dict, List, Tuple
from itertools import permutations
from functools import cmp_to_key
from z2 import Z2
import copy
import sys
from collections import OrderedDict
from sortedcontainers import SortedDict
from functools import total_ordering

NEG = 0b10;
POS = 0b01;
DISAGREE = 0b11;
AGREE = 0b00;
UNSET = 0b00;
BITS = 0b11;

def lex_order(first: np.ndarray[Any, np.dtype[Z2]], second:np.ndarray[Any, np.dtype[Z2]], first_sign_mask,second_sign_mask ) -> int:

    #set mask to 
    arr_length = first.shape[0]
    # mask = ((1 << arr_length) - 1)
    # mask = 0xFFFF
    # first_sign_mask = first_sign_mask & mask
    # second_sign_mask =  second_sign_mask & mask

    # print(bin(first_sign_mask))
    # print(bin(second_sign_mask))
    # print(bin(mask))
    # sys.exit(0)

    for i in range(arr_length):
        comp1 = (first[i].int_c > 0) - (first[i].int_c < 0)
        comp2 = (second[i].int_c > 0) - (second[i].int_c < 0)
        
        if comp1 == 0 and comp2 == 0:
            continue
        if comp1 == 0:
            return 1  # Greater
        if comp2 == 0:
            return -1  # Less
        
        fsm = (first_sign_mask  >> (2*i)) & BITS
        ssm = (second_sign_mask >> (2*i)) & BITS
        
        if (comp1 < 0) ^ (fsm == NEG):
            first_sign_mask = first_sign_mask ^ 0xFFFF
        if (comp2 < 0) ^ (ssm == NEG):
            second_sign_mask =second_sign_mask ^ 0xFFFF
        break

    for j in range(i,arr_length):

        first_is_neg = ((first_sign_mask  >> (2*j)) & BITS) == NEG
        second_is_neg = ((second_sign_mask  >> (2*j)) & BITS) == NEG
        
        first_val = (-first[j] if first_is_neg else first[j]).data()
        second_val = (-second[j] if second_is_neg else second[j]).data()

        comparison = (second_val > first_val) - (second_val < first_val)
        
        if comparison == 0:
            continue
        if first[j].int_c == 0:
            return 1  # Greater
        if second[j].int_c == 0:
            return -1  # Less
        
        return comparison
    
    return 0  # Equal


class SO6_basis:
    def __init__(self,arr = None,initialize='zero',args=None):
        """Initializes a 6x6 matrix with Z2 elements, defaulting to zero."""

        #data
        if arr is not None:
            self.arr = arr.copy()
        
        else:
            self.arr = np.array([[Z2() for _ in range(6)] for _ in range(6)])
            if initialize == 'random':
                for row in range(6):
                    for col in range(6):
                        self.arr[row,col]  = Z2(int_c=np.random.randint(-3, 3), 
                                                sqrt2_c=np.random.randint(-3, 3), 
                                                denom_exp=np.random.randint(4, 8))
                
            elif initialize == 'identity':
                for row in range(6):
                    self.arr[row,row] = Z2(1,0,0)
            elif initialize == 'circuit':
                for row in range(6):
                    self.arr[row,row] = Z2(1,0,0)
                self.add_from_circuit(args)

    def get_arr(self,type = (36,3)):
        if type == (36,3):
            new_arr  = np.zeros((36,3))
            for row in range(6):
                for col in range(6):
                    new_arr[col*6+row,:] = np.array ([self.arr[row,col].int_c,self.arr[row,col].sqrt2_c,self.arr[row,col].denom_exp])
            return new_arr
        else:
            return NotImplemented

    @classmethod
    def from_packed_array(cls, array):
        """
        Reconstruct an SO6 object from a (36, 3) packed array.
        """
        assert array.shape == (36, 3), "Expected shape (36, 3)"
        arr = np.empty((6, 6), dtype=object)
        for idx in range(36):
            col = idx // 6
            row = idx % 6
            int_c, sqrt2_c, denom_exp = array[idx]
            arr[row, col] = Z2(int(int_c), int(sqrt2_c), int(denom_exp), reduce_flag=True)
        return cls(arr=arr)

    def add_from_circuit(self,circuit: List):
        for c in circuit:
            self.left_multiply_by_T(c)
        
    def __repr__(self):
        return "\n".join([", ".join(map(str, row)) for row in self.arr])
    
    def reduce(self):
         for row in range(6):
            for col in range(6):
                self.arr[row,col].reduce()       

    def to_float(self):
        new_arr = np.zeros([6,6],dtype=np.float64)
        for row in range(6):
            for col in range(6):
                new_arr[row,col] = self.arr[row,col].to_float()
        return new_arr
    
    def get_lde(self):
        new_arr = np.zeros([6,6],dtype=np.float64)
        for row in range(6):
            for col in range(6):
                new_arr[row,col] = self.arr[row,col].denom_exp
        return int(np.max(new_arr))

    def get_exp(self):
        new_arr = np.zeros([6,6],dtype=np.float64)
        for row in range(6):
            for col in range(6):
                new_arr[row,col] = self.arr[row,col].denom_exp
        return new_arr      
 
      
    def left_multiply_by_T(self, i ,inplace = True):
        """Left multipliekl/ the matrix by a transformation matrix T_i."""
        pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5),
                 (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
        
        if i < 0 or i >= len(pairs):
            raise ValueError("Invalid value for i")
        
        row1, row2 = pairs[i]

        if inplace == True:
            for col in range(6):
                row1_element = copy.deepcopy(self.arr[row1,col]) 
                row2_element = copy.deepcopy(self.arr[row2,col])
     
                # update elements
                
                self.arr[row1,col] = row1_element + row2_element
                self.arr[row2,col] = row2_element - row1_element
    
                if self.arr[row1,col].int_c !=0 :
                    self.arr[row1,col].denom_exp = self.arr[row1,col].denom_exp + 1
    
                if self.arr[row2,col].int_c !=0 :
                    self.arr[row2,col].denom_exp = self.arr[row2,col].denom_exp + 1
    
            self.last_T = i
        else:
            new_arr = self.arr.copy()
            for col in range(6):  
                row1_element = copy.deepcopy(self.arr[row1,col]) 
                row2_element = copy.deepcopy(self.arr[row2,col])

                new_arr[row1,col] = row1_element + row2_element
                new_arr[row2,col] = row2_element - row1_element
    
                if new_arr[row1,col].int_c !=0 :
                    new_arr[row1,col].denom_exp = new_arr[row1,col].denom_exp + 1
    
                if new_arr[row2,col].int_c !=0 :
                    new_arr[row2,col].denom_exp = new_arr[row2,col].denom_exp + 1
            
            new_so6 = SO6(new_arr)
            new_so6.last_T = i
            return new_so6



    def left_multiply_by_inverse_T(self, i ,inplace = True):
        """Left multipliekl/ the matrix by a transformation matrix T_i."""
        pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5),
                 (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
        
        if i < 0 or i >= len(pairs):
            raise ValueError("Invalid value for i")
        
        row1, row2 = pairs[i]

        if inplace == True:
            for col in range(6):
                row1_element = copy.deepcopy(self.arr[row1,col]) 
                row2_element = copy.deepcopy(self.arr[row2,col])
     
                # update elements
                
                self.arr[row1,col] = row1_element - row2_element
                self.arr[row2,col] = row2_element + row1_element
    
                if self.arr[row1,col].int_c !=0 :
                    self.arr[row1,col].denom_exp = self.arr[row1,col].denom_exp + 1
    
                if self.arr[row2,col].int_c !=0 :
                    self.arr[row2,col].denom_exp = self.arr[row2,col].denom_exp + 1
    
            self.last_T = i
        else:
            new_arr = self.arr.copy()
            for col in range(6):  
                row1_element = copy.deepcopy(self.arr[row1,col]) 
                row2_element = copy.deepcopy(self.arr[row2,col])

                new_arr[row1,col] = row1_element - row2_element
                new_arr[row2,col] = row2_element + row1_element
    
                if new_arr[row1,col].int_c !=0 :
                    new_arr[row1,col].denom_exp = new_arr[row1,col].denom_exp + 1
    
                if new_arr[row2,col].int_c !=0 :
                    new_arr[row2,col].denom_exp = new_arr[row2,col].denom_exp + 1
            
            new_so6 = SO6(new_arr)
            new_so6.last_T = i
            return new_so6


@total_ordering
class SO6(SO6_basis):
    def __init__(self,arr = None,initialize='identity',args =None):
        """Initializes a 6x6 matrix with Z2 elements, defaulting to zero."""

        super().__init__(arr)

        #for canonical_form
        self.__row_frequency_map = [SortedDict() for _ in range(6)]
        self.__col_frequency_map = [SortedDict() for _ in range(6)]


        self.Col = [0,1,2,3,4,5]
        self.Row = [0,1,2,3,4,5]

        self.last_T = -1
        self.sign_convention = 0b010101010101
        self.col_mask = 0x3F


        #for future hash
        self.hash = 0

        self.circuit_list = []

        if arr is not None:
            self.canonical_form()

        elif initialize == 'identity':
            for row in range(6):
                self.arr[row,row] = Z2(1,0,0,reduce_flag=True)
            self.canonical_form()
                
        elif initialize == 'circuit':
            for row in range(6):
                self.arr[row,row] = Z2(1,0,0,reduce_flag=True)
            self.circuit_list = args
            self.add_from_circuit(args)
    
    def add_from_circuit(self,circuit: List):
        for c in circuit:
            self.left_multiply_by_T(c)
        self.canonical_form()
        
    def compute_hash(self):
        return None
   
        
    def __set_mask_sign(self,mask,index,sign):
        new_mask = mask^(((mask >> (2 * index)) & BITS) << (2 * index)) | (sign << (2 * index))
        return new_mask
    

    def get_perm_matrix(self):
        new_so6 = self.arr[self.Row, :][:, self.Col].copy()
        return SO6(new_so6)
    
    #only for test
    def print_perm_packed(self):
        new_arr = self.arr[self.Row, :][:, self.Col].copy()
        new_packed = np.zeros((6,6),dtype=np.int32)
        for row in range(6):
            for col in range(6):
                new_packed[row,col] = new_arr[row,col].to_packed()
        print('packed after permutation')
        print(new_packed)


    def get_arr_after_perm(self,type = (36,3)):
        if type == (36,3):
            new_arr  = np.zeros((36,3))
            for row in self.Row:
                for col in self.Col:
                    new_arr[col*6+row,:] = np.array ([self.arr[row,col].int_c,self.arr[row,col].sqrt2_c,self.arr[row,col].denom_exp])
            return new_arr
        else:
            return NotImplemented




    def canonical_form(self):
        self.__calculate_frequency_map()


        row_ecs = self.__row_equivalence_classes()
        col_ecs = self.__col_equivalence_classes()

        row_perm = []
        col_perm = []

        for _, rows in row_ecs.items():
            row_perm.extend(rows)
        
        for _, cols in col_ecs.items():
            col_perm.extend(cols)

        # print(row_perm)
        # print(col_perm)
        self.Row = row_perm.copy()
        self.Col = col_perm.copy()

        while True:
            temp_row_perm = []

            # temp_row_perm includes all rows because the construction of row_ecs
            for _, group in row_ecs.items():
                temp_row_perm.extend(group)


            #????
            for k in range(32):
                sc = POS      # utils::POS = 0b10
                for l in range(1, 6):
                    if k & (1 << (l-1)):
                        sc = self.__set_mask_sign(sc,l,NEG)
                    else:
                        sc = self.__set_mask_sign(sc,l,POS)
                
                def comparator(i, j):
                    left = (self.arr[temp_row_perm, :].T)[i]
                    right = (self.arr[temp_row_perm, :].T)[j]
                    comp = lex_order(left, right, sc, sc)
                    return comp 
                
                temp_col_perm = []
                for _, col_class in col_ecs.items():
                    col_class.sort(key=cmp_to_key(comparator))
                    temp_col_perm.extend(col_class)
                
                if self.__is_better_permutation(temp_row_perm, temp_col_perm, sc):
                    self.Row = np.array(temp_row_perm).copy()
                    self.Col = np.array(temp_col_perm).copy()
                    self.sign_convention = sc
            
            # for key, val in col_ecs.items():
            #     print(key , ':',val)

            # print()
            if not self.__get_next_equivalence_class(row_ecs):
                break

    def __is_better_permutation(self, row_perm, col_perm, sign_perm):

        temp_arr =  self.arr[row_perm, :][:, col_perm].T
        curr_arr =  self.arr[self.Row, :][:, self.Col].T
        for col in range(6):
            current = curr_arr[col]
            new_col = temp_arr[col]

            comparison = lex_order(current, new_col, self.sign_convention, sign_perm)

            if comparison == 0:
                continue
            return comparison == 1

        return False
    
    def __calculate_frequency_map(self):

        self.__row_frequency_map = [SortedDict() for _ in range(6)]
        self.__col_frequency_map = [SortedDict() for _ in range(6)]

        for row in range(6):
            for col in range (6):
                key = abs(self.arr[row,col]).data()
                self.__row_frequency_map[row][key] = self.__row_frequency_map[row].get(key, 0) + 1
                self.__col_frequency_map[col][key] = self.__col_frequency_map[col].get(key, 0) + 1


    def __row_equivalence_classes(self):
        ret = SortedDict()
        for row in range(6):
            key = tuple(sorted(self.__row_frequency_map[row].items()))
            # print(self.__row_frequency_map[row])
            # print(key)
            # sys.exit(0)
            if key not in ret.keys():
                ret[key] = [row]
            else:
                ret[key].append(row)
        return ret


    def __col_equivalence_classes(self):
        ret = SortedDict()
        for col in range(6):
            key = tuple(sorted(self.__col_frequency_map[col].items()))
            # print(self.__row_frequency_map[col])
            # print(key)
            if key not in ret.keys():
                ret[key] = [col]
            else:
                ret[key].append(col)
        return ret


    def __get_next_equivalence_class(self,row_equivalence_classes):
    
        more_permutations = False

        # key is the frequency map, group is the list of rows
        for key, group in row_equivalence_classes.items():

            # Convert group to list of permutations
            perms = sorted(permutations(group))
            # Generate all sorted permutations
            # Find current position
            current_index = perms.index(tuple(group))
            next_index = current_index + 1

            if next_index < len(perms):
                row_equivalence_classes[key] = list(perms[next_index])  # Set next permutation
                more_permutations = True
                break
            else:
                row_equivalence_classes[key] = sorted(group)  # Reset to sorted order
    
        return more_permutations
    
    def left_mul(self,other):
        if not isinstance(other, SO6):  # Ensure 'other' is an instance of MyClass
            return NotImplemented  
        new_arr = self.arr.copy()
        new_arr = np.matmul(other.arr,new_arr)
        return SO6(new_arr)
    
    def transpose(self):
        new_arr = self.arr.copy()
        new_arr = new_arr.T
        return SO6(new_arr)
    
    def __lt__(self, other):
        if not isinstance(other, SO6):  # Ensure 'other' is an instance of MyClass
            return NotImplemented  
        first_arr =  self.arr[self.Row, :][:, self.Col].T
        other_arr =  other.arr[other.Row, :][:, other.Col].T

        for col in range(6):
            first_list = first_arr[col]
            other_list = other_arr[col]
            comparison = lex_order(first_list, other_list, self.sign_convention, other.sign_convention)
            if comparison !=0:
                return comparison < 0

        return False

    def __eq__(self, other):
        if not isinstance(other, SO6):  # Ensure 'other' is an instance of MyClass
            return NotImplemented  
        first_arr =  self.arr[self.Row, :][:, self.Col].T
        other_arr =  other.arr[other.Row, :][:, other.Col].T

        for col in range(6):
            first_list = first_arr[col]
            other_list = other_arr[col]
            comparison = lex_order(first_list, other_list, self.sign_convention, other.sign_convention)
            if comparison !=0:
                return False

        return True

