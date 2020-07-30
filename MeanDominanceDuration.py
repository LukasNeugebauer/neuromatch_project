#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:59:14 2020

@author: kristinkaduk
"""


##### 
import numpy as np
import matplotlib.pyplot as plt
from network import Network
from defaults import get_input, get_default_parameters


    params = get_default_parameters()

    # initialize the network
    network = Network(.5, *params)
    # print(network.snapshot)

    dt = params[0]['dt']
    total_duration = params[0]['total_duration']
    sensory_input = get_input(dt, total_duration, contrast = np.array([[.5, 0.], [0., .5]]))

    sensory_input = sensory_input[:30000]

    timecourse = network.simulate(sensory_input)
    
    
######
    # average duration of all of the individual periods for which one of the rivalry stimuli dominates)

def DominanceDuration(timecourse, params, visualization =1):      
    
    
    
    Time = params[0]['time_vector']
    Summation1 = timecourse.get_neuron_over_time('summation_1')
    Summation2 = timecourse.get_neuron_over_time('summation_2')

    # activity for Summation is bigger -> 1
    Dominance_SumNeuron =  Summation2 > Summation1

    print('How many changes in dominance? ')
    print(np.sum([np.abs(np.diff(Dominance_SumNeuron)) == 1]))
    
    idx_start = np.where([np.diff(Dominance_SumNeuron*1) == -1])[1]
    idx_end = np.where([np.diff(Dominance_SumNeuron*1) == 1])[1]
    idx = np.sort(np.concatenate([idx_start ,idx_end]))
    
    if visualization == 1: # visualization
        fig
        plt.plot( Dominance_SumNeuron, 'r')
        plt.plot(Summation2)
        plt.plot( idx, np.zeros(np.shape(idx)), 'ko')
    
    a = np.arange(np.shape(idx)[0])
    idx_odd = [num for num in a if num % 2 == 1]   
    idx_even = [num for num in a if num % 2 == 0]  
    
    
    DominanceDuration_SumNeuron2 =  Time[idx[idx_odd]] - Time[idx[idx_even[:-1]]]
    DominanceDuration_SumNeuron1 =  Time[idx[idx_even[1:]]] - Time[idx[idx_odd] ]
    
    PercentageDominance_SumNeuron2 = (np.sum(Dominance_SumNeuron == 1))/np.size(Dominance_SumNeuron)

    return DominanceDuration_SumNeuron2, DominanceDuration_SumNeuron1
    
# visualisation function
# AIM: A graph which 
    
AttentionWeight = params[0]['weight_attention']
x = np.arange(np.shape(y)[0])+1
# Matrix - each row is a different computation
y = [DominanceDuration_SumNeuron2.mean(), DominanceDuration_SumNeuron1.mean()]
group = 


def plot_GroupComparison(x, y, group):    
    
    
    fig
    plt.plot(  x, y,  'o', ,
          label=group)
    plt.set_ylim([0.,np.max(y) + np.std(y)])
    
    return 
 





    
    
    