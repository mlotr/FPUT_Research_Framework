# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:54:08 2020

@author: Tiziana Comito
"""
import numpy as np
import operator

def w_fun(k):
    '''
    Parameters
    ----------
    k : array-like or scalar
        The funtion requires an array of wavenumbers and
        return the dispersion relation for the FPUT system.

    Returns
    -------
    Omega is an array like / scalar (depending from the imput) which
    represent the frequency of the oscillation of the system in linear 
    approximation.

    '''
    pi = np.pi
    N = len(k)
    omega =2*np.abs( np.sin(pi*(k)/N))
    #omega =2*np.sqrt(K/m)*np.sin(pi*(k)/N)
    return(omega)


# Linear combination possible
def fun3_1(o1,o2,o3):
    return(+o1 -o2 -o3)
def fun3_2(o1,o2,o3):
    return(+o1 +o2 -o3 )
def fun3_3(o1,o2,o3):
    return(+o1 +o2 +o3 )

def fun4_1(o1,o2,o3,o4):
    return(o1 -o2 -o3 - o4)
def fun4_2(o1,o2,o3,o4):
    return(o1 +o2 -o3 - o4)
def fun4_3(o1,o2,o3,o4):
    return(o1 +o2 +o3 - o4)
def fun4_4(o1,o2,o3,o4):
    return(o1 +o2 +o3 + o4)

def fun5_1(o1,o2,o3,o4,o5):
    return(o1 -o2 -o3 - o4 - o5)
def fun5_2(o1,o2,o3,o4,o5):
    return(o1 +o2 -o3 - o4 - o5)
def fun5_3(o1,o2,o3,o4,o5):
    return(o1 +o2 +o3 - o4 - o5)
def fun5_4(o1,o2,o3,o4,o5):
    return(o1 +o2 +o3 + o4 - o5)
def fun5_5(o1,o2,o3,o4,o5):
    return(o1 +o2 +o3 + o4 + o5)

#%%

def funM_j(list_o,j):
    
    M = len(list_o)
    
    op1 = [operator.sub, operator.sub, operator.sub, operator.sub]
    op2 = [operator.add, operator.sub, operator.sub, operator.sub]
    op3 = [operator.add, operator.add, operator.sub, operator.sub]
    op4 = [operator.add, operator.add, operator.add, operator.sub]
    op5 = [operator.add, operator.add, operator.add, operator.add]
    
    operation_dic = {1:op1,2:op2,3:op3,4:op4,5:op5}
    
    op = operation_dic[j]
    s = 0
    
    for m in range(M-1):
        
        if m ==0:
            s = op[0](list_o[0],list_o[1])
        else:
            s = op[m](s,list_o[m+1])

    return s

#%%
# Linear combination possible
def fun3_1_w(o1,o2,o3,w):
    return(np.round(+w[o1] -w[o2] -w[o3],15) )
def fun3_2_w(o1,o2,o3,w):
    return(np.round(+w[o1] +w[o2] -w[o3],15) )
def fun3_3_w(o1,o2,o3,w):
    return(np.round(+w[o1] +w[o2] +w[o3],15))

def fun4_1_w(o1,o2,o3,o4,w):
    return(np.round(w[o1] -w[o2] -w[o3] - w[o4],15))
def fun4_2_w(o1,o2,o3,o4,w):
    return(np.round(w[o1] +w[o2] -w[o3] - w[o4],15))
def fun4_3_w(o1,o2,o3,o4,w):
    return(np.round(w[o1] +w[o2] +w[o3] - w[o4],15))
def fun4_4_w(o1,o2,o3,o4,w):
    return(np.round(w[o1] +w[o2] +w[o3] + w[o4],15))

def fun5_1_w(o1,o2,o3,o4,o5,w):
    return(np.round(w[o1] -w[o2] -w[o3] - w[o4] - w[o5],15))
def fun5_2_w(o1,o2,o3,o4,o5,w):
    return(np.round(w[o1] +w[o2] -w[o3] - w[o4] - w[o5],15))
def fun5_3_w(o1,o2,o3,o4,o5,w):
    return(np.round(w[o1] +w[o2] +w[o3] - w[o4] - w[o5],15))
def fun5_4_w(o1,o2,o3,o4,o5,w):
    return(np.round(w[o1] +w[o2] +w[o3] + w[o4] - w[o5],15))
def fun5_5_w(o1,o2,o3,o4,o5,w):
    return(np.round(w[o1] +w[o2] +w[o3] + w[o4] + w[o5],15))


def delta_k(k,n,resonance_type):
    '''
    Parameters
    ----------
    k : array-like.
        Vector or list containing the wavenumbers.
    n : integer in (3,5).
        A value expressing the number of waves involved.
    resonance_type : integer from 1 to 5, with condition resonance_type <= n
        The argument takes a value from 1 to 5 with respect the type of
        linear combination whated. Notice that the different types of LC
        available are limited and sequentially ordered in the following way:
        
        - the first element of the combination is always positive;
        - you have a number of '+' = to resonance_type
        - (Commutative property of linear algebra). The combination is structured
        in such a way that you first have in order all the positive contributions
        and then the negatives, e.g. if n=5 and resonace_type=3 gives a 
        combination with 3 '+' and two '-':
        +o1 +o2 +o3 -o4 -o5, where o_i represent the i-th object.

    Returns
    -------
    A n-dim Tensor made of 0 and 1 generalizing the Kronacker-delta.

    '''
    N = len(k)

    tensor_k = np.zeros([N]*n)

    if n == 3:
        for i1 in range(1,N):                 
             for i2 in range(1,N):         
                 for i3 in range(1,N):


                    delta1 = fun3_1(i1,i2,i3)%N
                    delta2 = fun3_2(i1,i2,i3)%N
                    delta3 = fun3_3(i1,i2,i3)%N
                    
                    op_dic ={1:delta1,2:delta2,3:delta3}
                    op = op_dic[resonance_type]
                    
                    if op == 0:
                        tensor_k[i1][i2][i3] = 1
                    
    elif n == 4:

        for i1 in range(1,N):                 
             for i2 in range(1,N):         
                 for i3 in range(1,N):
                     for i4 in range(1,N):    
                         delta1 = fun4_1(i1,i2,i3,i4)%N
                         delta2 = fun4_2(i1,i2,i3,i4)%N
                         delta3 = fun4_3(i1,i2,i3,i4)%N
                         delta4 = fun4_4(i1,i2,i3,i4)%N 

                         op_dic ={1:delta1,2:delta2,3:delta3,4:delta4}
                         op = op_dic[resonance_type]
                      
                         if op == 0:
                              tensor_k[i1][i2][i3][i4] = 1
    elif n == 5:

        for i1 in range(1,N):                 
             for i2 in range(1,N):         
                 for i3 in range(1,N):
                     for i4 in range(1,N):  
                         for i5 in range(1,N):  
                             delta1 = fun5_1(i1,i2,i3,i4,i5)%N
                             delta2 = fun5_2(i1,i2,i3,i4,i5)%N
                             delta3 = fun5_3(i1,i2,i3,i4,i5)%N
                             delta4 = fun5_4(i1,i2,i3,i4,i5)%N 
                             delta5 = fun5_5(i1,i2,i3,i4,i5)%N 
                             
                             op_dic ={1:delta1,2:delta2,3:delta3,4:delta4, 5:delta5}
                             op = op_dic[resonance_type]
                      
                             if op == 0:
                                  tensor_k[i1][i2][i3][i4][i5] = 1
    else:
        print("This function takes n=3,4,5") 
        return                 
    return(tensor_k)

def delta_w(k,n,resonance_type):
    '''
    Parameters
    ----------
    k : array-like.
        Vector or list containing the wavenumbers.
    n : integer in (3,5).
        A value expressing the number of waves involved.
    resonance_type : integer from 1 to 5, with condition resonance_type <= n
        The argument takes a value from 1 to 5 with respect the type of
        linear combination whated. Notice that the different types of LC
        available are limited and sequentially ordered in the following way:
        
        - the first element of the combination is always positive;
        - you have a number of '+' = to resonance_type
        - (Commutative property of linear algebra). The combination is structured
        in such a way that you first have in order all the positive contributions
        and then the negatives, e.g. if n=5 and resonace_type=3 gives a 
        combination with 3 '+' and two '-':
        +o1 +o2 +o3 -o4 -o5, where o_i represent the i-th object.

    Returns
    -------
    A n-dim Tensor made of 0 and 1 generalizing the Kronacker-delta.

    '''
    N = len(k)
    w = w_fun(k)
    tensor_w = np.zeros([N]*n)

    if n == 3:
        for i1 in range(1,N):                 
             for i2 in range(1,N):         
                 for i3 in range(1,N):


                    delta1_w = fun3_1_w(i1,i2,i3,w)
                    delta2_w = fun3_2_w(i1,i2,i3,w)
                    delta3_w = fun3_3_w(i1,i2,i3,w)
                    
                    op_dic_w ={1:delta1_w,2:delta2_w,3:delta3_w}
                    op = op_dic_w[resonance_type]
                    
                    if np.abs(op) <= 10**(-15):
                        tensor_w[i1][i2][i3] = 1
                    
    elif n == 4:

        for i1 in range(1,N):                 
             for i2 in range(1,N):         
                 for i3 in range(1,N):
                     for i4 in range(1,N):    
                         delta1_w = fun4_1_w(i1,i2,i3,i4,w)
                         delta2_w = fun4_2_w(i1,i2,i3,i4,w)
                         delta3_w = fun4_3_w(i1,i2,i3,i4,w)
                         delta4_w = fun4_4_w(i1,i2,i3,i4,w)

                         op_dic_w ={1:delta1_w,2:delta2_w,3:delta3_w,4:delta4_w}
                         op = op_dic_w[resonance_type]
                      
                         if np.abs(op) <= 10**(-15):
                              tensor_w[i1][i2][i3][i4] = 1
    elif n == 5:

        for i1 in range(1,N):                 
             for i2 in range(1,N):         
                 for i3 in range(1,N):
                     for i4 in range(1,N):  
                         for i5 in range(1,N):  
                             delta1_w = fun5_1_w(i1,i2,i3,i4,i5,w)
                             delta2_w = fun5_2_w(i1,i2,i3,i4,i5,w)
                             delta3_w = fun5_3_w(i1,i2,i3,i4,i5,w)
                             delta4_w = fun5_4_w(i1,i2,i3,i4,i5,w) 
                             delta5_w = fun5_5_w(i1,i2,i3,i4,i5,w) 
                             
                             op_dic_w ={1:delta1_w,2:delta2_w,3:delta3_w,4:delta4_w, 5:delta5_w}
                             op = op_dic_w[resonance_type]
                      
                             if np.abs(op) <= 10**(-15):
                                  tensor_w[i1][i2][i3][i4][i5] = 1
    else:
        print("This function takes n=3,4,5") 
        return                 
    return tensor_w 

