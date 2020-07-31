"""
Reproduce figure 2 of the original publication and possible more
"""

from network import Network
from utilities import timecourse2pandas, add_zero_startpoint, get_model_output
from defaults import get_default_parameters, get_input
from os.path import join, exists
import os
import numpy as np
import matplotlib.pyplot as plt
from plotting import savefig


    
def reproduce_figure_2(**fig_kwargs):
    """
    specifically tuned to get figure 2
    """
    #attended, dichoptic
    print('Attended, dichoptic')
    network_params = get_default_parameters()
    network_params = add_zero_startpoint(network_params)
    input_params = {
        'dt': network_params[0]['dt'],
        'total_duration': network_params[0]['total_duration']
    }
    df_attended_dichopt = get_model_output(network_params, input_params)
    sum_1_attended_dichopt = df_attended_dichopt['summation_1']
    sum_2_attended_dichopt = df_attended_dichopt['summation_2']
    
    #unattended, dichoptic
    print('Unattended, dichoptic')
    network_params[0].update({'weight_attention': 0})
    df_unattended_dichopt = get_model_output(network_params, input_params)
    sum_1_unattended_dichopt = df_unattended_dichopt['summation_1']
    sum_2_unattended_dichopt = df_unattended_dichopt['summation_2']
    
    #attended, monocular
    print('Attended, monocular')
    network_params = get_default_parameters()
    network_params = add_zero_startpoint(network_params)
    input_params = {
        'dt': network_params[0]['dt'],
        'total_duration': network_params[0]['total_duration'],
        'contrast': np.array([[.5, .5],[0, 0]])
    }
    df_attended_monoc = get_model_output(network_params, input_params)
    sum_1_attended_monoc = df_attended_monoc['summation_1']
    sum_2_attended_monoc = df_attended_monoc['summation_2']
    
    #unattended, monocular
    print('Unattended, monocular')
    network_params[0].update({'weight_attention': 0})
    df_unattended_monoc = get_model_output(network_params, input_params)
    sum_1_unattended_monoc = df_unattended_monoc['summation_1']
    sum_2_unattended_monoc = df_unattended_monoc['summation_2']
    
    #prepare the figure
    fig, axs = plt.subplots(
        nrows=2, 
        ncols=2, 
        figsize=(30,10), 
        sharex=True, 
        sharey=True, 
        constrained_layout=True
    )
    
    n_iter = network_params[0]['time_vector'].size
    dt     = network_params[0]['dt']
    xticks = np.arange(0,n_iter + 1, 5000 // dt)
    xticklabels = [str(int(x * dt / 1000)) for x in xticks]
    axs[0,0].plot(sum_1_attended_dichopt, 'g', label='+45°', linewidth=4)
    axs[0,0].plot(sum_2_attended_dichopt, 'b', label='-45°', linewidth=4)
    axs[0,0].set_ylim(0,1)
    axs[0,0].set_yticks([0, .5, 1])
    axs[0,0].set_xticks(xticks)
    axs[0,0].set_xlim(0)
    axs[0,0].set_ylabel('Response', fontsize=30)
    axs[0,0].set_title('Attended', fontsize=30)
    axs[0,0].tick_params(axis='both', which='major', labelsize=30)
    axs[0,0].spines['right'].set_visible(False)
    axs[0,0].spines['top'].set_visible(False)

    axs[1,0].plot(sum_1_unattended_dichopt, 'g', label='+45°', linewidth=4)
    axs[1,0].plot(sum_2_unattended_dichopt, 'b', label='-45°', linewidth=4)
    axs[1,0].set_ylim(0,1)
    axs[1,0].set_yticks([0, .5, 1])
    axs[1,0].set_xticks(xticks)
    axs[1,0].set_xticklabels(xticklabels)
    axs[0,0].set_xlim(0)
    axs[1,0].set_ylabel('Response', fontsize=30)
    axs[1,0].set_xlabel('Time (sec)', fontsize=30)
    axs[1,0].set_title('Unattended', fontsize=30)
    axs[1,0].tick_params(axis='both', which='major', labelsize=30)
    axs[1,0].spines['right'].set_visible(False)
    axs[1,0].spines['top'].set_visible(False)
    
    axs[0,1].plot(sum_1_attended_monoc, 'g', label='+45°', linewidth=4)
    axs[0,1].plot(sum_2_attended_monoc, 'b', label='-45°', linewidth=4)
    axs[0,1].set_title('Attended', fontsize=30)
    axs[0,1].spines['right'].set_visible(False)
    axs[0,1].spines['top'].set_visible(False)
    
    axs[1,1].plot(sum_1_unattended_monoc, 'g', label='+45°', linewidth=4)
    axs[1,1].plot(sum_2_unattended_monoc, 'b', label='-45°', linewidth=4)
    axs[1,1].set_xlabel('Time (sec)', fontsize=30)
    axs[1,1].set_title('Unattended', fontsize=30)
    axs[1,1].set_xticks(xticks)
    axs[1,1].set_xticklabels(xticklabels)
    axs[1,1].tick_params(axis='both', which='major', labelsize=30)
    axs[1,1].spines['right'].set_visible(False)
    axs[1,1].spines['top'].set_visible(False)
    
    return fig, axs
    
if __name__ == '__main__':
    fig, axs = reproduce_figure_2()
    folder = '/home/lukas/Documents/education/neuromatch_summer_school/project/plots'
    if not exists(folder):
        os.mkdir(folder)
    name = 'plot_2.png'
    savefig(fig, name, folder)