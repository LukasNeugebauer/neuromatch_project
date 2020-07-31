"""
Plotting functions to visualize output from network
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
from utilities import timecourse2pandas
from os.path import join


def plot_cmp_dominance(
    x_ticks, 
    duration_1, 
    duration_2, 
    dominance, 
    xlabel='Difference in input to attention neurons', 
    title='Influence of biasing attention neurons'
):
    assert x_ticks.size == duration_1.size == duration_2.size == dominance.size
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25, 10), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=40)
    
    axs[0].plot(x_ticks, duration_1, 'g-', linewidth=4)
    axs[0].plot(x_ticks, duration_2, 'b-', linewidth=4)
    axs[0].set_xlabel(xlabel, fontsize=35)
    axs[0].set_ylabel('Average duration of dominance (ms)', fontsize=35)
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels([str(x) for x in x_ticks])
    axs[0].tick_params(axis='both', which='major', labelsize=25)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    
    axs[1].plot(x_ticks, dominance, 'k-', linewidth=4)
    axs[1].set_xlabel(xlabel, fontsize=35)
    axs[1].set_ylabel('Ratio of dominance periods', fontsize=35)
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels([str(x) for x in x_ticks])
    axs[1].tick_params(axis='both', which='major', labelsize=25)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    
    return fig, axs
    
    

def plot_timecourse(
    timecourse,
    dt=1.
):
    neuron_classes = ['sensory', 'summation', 'opponency', 'attention']
    legend_keys = {
        'left_1': 'left eye, -45°',
        'right_1': 'right eye, -45°',
        'left_2': 'left eye, +45°',
        'right_2': 'right eye, +45°',
        '1': '+45°',
        '2': '-45°',
        'left': 'left eye',
        'right': 'right eye'
    }
    label_size = max([len(x) for x in legend_keys.values()])
    df = timecourse2pandas(timecourse)
    #xtick-stuff
    n_samples = timecourse.get_neuron_over_time(f'sensory_left_1').size
    x = np.arange(n_samples) * (dt)
    xticks = x[(x % (5000 / dt)) == 0]
    xticklabels = [str(5 * (i + 1)) for i in range(xticks.size)]
    
    #collect activities
    all_activities = {key: {} for key in neuron_classes}
    #sensory
    for eye, orient in product(['left', 'right'], ['1', '2']):
        all_activities['sensory']['_'.join([eye, orient])] = df[f'sensory_{eye}_{orient}']
    #summation 
    for orient in ['1', '2']:
        all_activities['summation'][orient] = df[f'summation_{orient}']
    #opponency
    for eye in ['left', 'right']:
        all_activities['opponency'][eye] = df[[f'opponency_{eye}_{orient}' for orient in [1, 2]]].sum(axis=1)
    #attention
    for orient in ['1', '2']:
        all_activities['attention'][orient] = df[f'attention_{orient}']
        
    #plotting
    fig, axs = plt.subplots(
        nrows=len(neuron_classes), 
        ncols=1, 
        sharex=True, 
        constrained_layout=True,
        figsize=(20,20)
    )
    for i, (typ, act) in enumerate(all_activities.items()):
        if typ == 'sensory':
            colors = np.array([
                [0, 180, 77],
                [230, 133, 36],
                [230, 230, 36],
                [75, 122, 191]
            ]) / 255.
        else:
            colors = np.array([
                [0, 180, 77],
                [75, 122, 191]
            ]) / 255.
            
        for j, (key, val) in enumerate(act.items()):
            axs[i].plot(
                x, val, color=colors[j], linewidth=4, label=legend_keys[key].ljust(label_size)
            )
        axs[i].set_xticks(xticks)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].legend(
            loc='center', 
            bbox_to_anchor=(1.25,.5), 
            fontsize=30,
            frameon=False
        )
        axs[i].set_title(f'{typ.capitalize()} neurons', fontsize=30)
        yticks = axs[i].get_yticks()
        y_range = np.arange(yticks.min(), yticks.max(), .05)
        y_range = (y_range * 100).round() / 100
        yticks = y_range[(y_range % .5) == 0]
        axs[i].set_yticks(yticks)
        axs[i].set_yticklabels([str(y) for y in yticks])
        axs[i].tick_params(axis='both', which='major', labelsize=25)
    
    axs[-1].set_xticklabels(xticklabels)
    
    return fig, axs
    

def savefig(fig, name, folder='', **kwargs):
    if folder:
        filename = join(folder, name)
    fig.savefig(filename, **kwargs)
    