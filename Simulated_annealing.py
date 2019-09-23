import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy
# -*- coding: utf-8 -*-
import sys

# Print iterations progress
def print_progress (iteration, total, prefix = '', suffix = '', decimals = 1, length = 30, fill = '%'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

"""
This program use simulated annealing to solve for the minima of the function

Model
    - Each state has energy E (we want to minimized this)
    - As time pass we keep increase and lower the temperature to ensure
    reaching the global minimum
    - Use metropolis algorithm to find equilibrium
"""
def Simulated_Annealing(state, lamda, T_c, epochs, T_init):
    E = []
    E.append(Energy(state))
    for epoch in range(epochs):
        T = T_init
        print('\nEpoch{0}/{1}\nT = {2:3f}\tE = {3:3f}'.format(epoch + 1, epochs, T, E[-1]))
        while(T > T_c):
            state, E_changed = MCstep(state, T, E[-1])
            E.append(E_changed)
            print_progress(1/T, 1/T_c)
            T = annealing(T, lamda)
    print()
    return state, E

"""
Use Metropolis algorithm to see whether the new state is going to be changed
"""
def MCstep(state, T, E_old) :

    new_state = swap(state)

    E_new = Energy(new_state)

    #Metropolis algorithm
    if E_new < E_old:
        return new_state, E_new
    elif random.random() < math.exp(-(E_new-E_old)/T):
        return new_state, E_new
    else :
        return state, E_old

"""
How to decrease temperature
"""

def annealing(T, lamda):
    return T*lamda



"""
Travelling Salesmen Problem

state - sequence of position [[x1,y1],[x2,y2],...[xn,yn]]
change - swap two position
"""
def Energy(state) :
    E = 0
    for i in range(len(state)):
        E += np.sum((state[i,:] - state[i-1,:])**2)
    return E

def swap(state):
    x = random.randint(0, len(state) -1)
    y = random.randint(0, len(state) -1)

    new_state = copy.deepcopy(state)

    new_state[x,:], new_state[y,:] = copy.deepcopy(new_state[y,:]), copy.deepcopy(new_state[x,:])

    return new_state

""" Main Function """
if __name__ == '__main__' :
    plt.figure(1)
    plt.title('Initial')
    st = np.random.rand(10,2)
    plt.plot(st[:,0], st[:,1], marker ='o', linestyle='-')
    new_st, E = Simulated_Annealing(st, 0.99, 0.001, 3, 100)
    plt.figure(2)
    plt.title('Final')
    plt.plot(new_st[:,0], new_st[:,1], marker ='o', linestyle='-')
    f = open('output', 'w')
    f.write('E = {}\n'.format(E[-1]))
    f.write('x\t\t\ty\n')
    for i in range(len(new_st)):
        f.write('{0}\t{1}\n'.format(new_st[i,0], new_st[i,1]))
    f.close()
    plt.figure(3)
    plt.plot(E)
    plt.xlabel('n')
    plt.ylabel('Energy')
    plt.show()
