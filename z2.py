import struct
import numpy as np
import math
from typing import Any, Dict, List, FrozenSet, Tuple
from itertools import permutations
from functools import cmp_to_key

class Z2:
    def __init__(self, int_c=0, sqrt2_c=0, denom_exp=0,reduce_flag = True):
        self.int_c = int_c  # Integer coefficient
        self.sqrt2_c = sqrt2_c  # sqrt(2) coefficient
        self.denom_exp = denom_exp  # Exponent for denominator
        self.reduce_flag = reduce_flag
    
    def set_value(self, int_c, sqrt2_c, denom_exp):
        self.int_c = int_c
        self.sqrt2_c = sqrt2_c
        self.denom_exp = denom_exp

    def __repr__(self):
        if self.denom_exp ==0:
            if self.sqrt2_c < 0:
                return f"({self.int_c} {self.sqrt2_c} √2)"
            else:
                return f"({self.int_c} + {self.sqrt2_c} √2)"
        else:
            if self.sqrt2_c < 0:
                return f"({self.int_c} {self.sqrt2_c} √2)/ √2^{self.denom_exp}"
            else:                          
                return f"({self.int_c} + {self.sqrt2_c} √2)/ √2^{self.denom_exp}"
            

    def to_float(self):
        """Convert the stored number to a floating-point representation."""
        return (self.int_c + self.sqrt2_c * (2 ** 0.5)) / (2 ** (self.denom_exp/2))

    def from_packed(self, packed_value):
        """Unpack from a 32-bit integer representation."""
        self.int_c = (packed_value & 0xFF) 
        self.sqrt2_c = ((packed_value >> 8) & 0xFF)
        self.denom_exp = (packed_value >> 16) 

    def to_packed(self):
        """Pack the structure into a 32-bit integer."""
        packed = (self.int_c & 0xFF) | ((self.sqrt2_c & 0xFF) << 8) | ((self.denom_exp & 0xFFFF) << 16)
        return packed
    
    def data(self):
        return self.to_packed() * int(self.int_c !=0)
    

    def __add__(self, other):

        if not isinstance(other, Z2):
            return NotImplemented
        if self.denom_exp == other.denom_exp:

            new_z2 = Z2( self.int_c + other.int_c, self.sqrt2_c + other.sqrt2_c, other.denom_exp)
            if self.reduce_flag == True:
                new_z2.reduce()
                return new_z2
            return new_z2
        
        else:
            exp_diff = self.denom_exp -  other.denom_exp

            if exp_diff >0:
                if exp_diff %2 ==0:
                    factor =  1 <<  int(exp_diff /2)
                    new_z2 = Z2( self.int_c + other.int_c * factor, self.sqrt2_c + other.sqrt2_c *factor, self.denom_exp)

                else:
                    factor1 =  1 <<  int((exp_diff+1) /2)
                    factor2 =  1 <<  int((exp_diff-1) /2)
                    new_z2 = Z2( self.int_c + other.sqrt2_c * (factor1), self.sqrt2_c + other.int_c *factor2, self.denom_exp)
                    
            else:
                exp_diff = - exp_diff
                if exp_diff %2 ==0:
                    factor =  1 <<  int(exp_diff /2)
                    new_z2 = Z2( self.int_c * factor + other.int_c , self.sqrt2_c *factor + other.sqrt2_c , other.denom_exp)
                else:
                    factor1 =  1 <<  int((exp_diff+1) /2)
                    factor2 =  1 <<  int((exp_diff-1) /2)
                    new_z2 = Z2( self.sqrt2_c* (factor1) + other.int_c , self.int_c *factor2 + other.sqrt2_c , other.denom_exp)

            if self.reduce_flag == True:
                new_z2.reduce()
                return new_z2
            return new_z2


    def __mul__(self, other):
        if not isinstance(other, Z2):
            return NotImplemented
        new_int_c = self.int_c * other.int_c + (self.sqrt2_c * other.sqrt2_c << 1)
        new_sqrt2_c = self.int_c * other.sqrt2_c + self.sqrt2_c * other.int_c
        new_denom_exp = self.denom_exp + other.denom_exp
        new_z2 = Z2(new_int_c, new_sqrt2_c, new_denom_exp)
        if self.reduce_flag == True:
            new_z2.reduce()
            return new_z2
        return new_z2

    
    def __neg__(self):
        return Z2(-self.int_c, -self.sqrt2_c, self.denom_exp)    

    def __sub__(self, other):
        return self + (-other)

    def __abs__(self):
        return Z2(abs(self.int_c),abs(self.sqrt2_c),self.denom_exp)
    
    def __eq__(self, other):
        return self.int_c == other.int_c and self.sqrt2_c == other.sqrt2_c and self.denom_exp == other.denom_exp
    



    def reduce(self):
        # print('self:',self)
        if self.int_c ==0 and self.sqrt2_c ==0:
            self.denom_exp =0 
            return
        
        if self.int_c %2 == 1:
            pass
        else: #self.int_c %2 == 0
            int_divisor=  (self.int_c & 0xFFFF) & ( (-self.int_c)&0xFFFF)
            sqrt_divisor = (self.sqrt2_c & 0xFFFF) & ( (-self.sqrt2_c)&0xFFFF)
            int_exp = (int_divisor.bit_length()-1) *2 
            sqrt_exp = (sqrt_divisor.bit_length() ) *2  -1 
            if int_exp <0 :
                min_exp = sqrt_exp
            elif sqrt_exp <0 :
                min_exp = int_exp
            else:
                min_exp = min(sqrt_exp,int_exp)

            if min_exp%2 ==0:
                right_shift = int(min_exp/2)
                new_int_c = self.int_c >> right_shift
                new_sqrt2_c = self.sqrt2_c  >> right_shift
                new_denom_exp = self.denom_exp - min_exp
            else:
                right_shift1 =  int((min_exp -1) /2)
                right_shift2 =  int((min_exp +1) /2)
                new_int_c = self.sqrt2_c >> right_shift1
                new_sqrt2_c = self.int_c >> right_shift2
                new_denom_exp = self.denom_exp - min_exp
            self.set_value(new_int_c,new_sqrt2_c,new_denom_exp)


