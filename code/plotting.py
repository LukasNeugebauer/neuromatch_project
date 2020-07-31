"""
Plotting functions to visualize output from network
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
from utilities import timecourse2pandas
from os.path import join

from glob import glob
import pickle
from os import path


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
    
    

def dominance_duration(timecourse, dt=0.5, visualization=1, file_base=''):
    # I adapted this for my dominance-plot function. It was an ugly fix for my stuff:/
    summation1 = timecourse.get_neuron_over_time('summation_1')
    summation2 = timecourse.get_neuron_over_time('summation_2')

    # activity for Summation is bigger -> 1
    dominance_bool = summation2 > summation1

    flanks = np.diff(dominance_bool)
    dom_switch_ids = np.where(abs(flanks))[0]

    try:
        if dom_switch_ids[0] < 100:
            dom_switch_ids = dom_switch_ids[1:]
        dom_periods = dt * np.diff(dom_switch_ids)
        dom_period_values = 1*(dominance_bool[dom_switch_ids+1])[:dom_periods.size]

        try:
            duration_0 = dom_periods[dom_period_values == 0].mean()
            duration_1 = dom_periods[dom_period_values == 1].mean()

            if np.isnan(duration_0) or np.isnan(duration_1):
                raise (Exception(''))
            #
        except:
            raise(Exception(''))
    except:
        if dominance_bool[-1] > 0:
            duration_0 = 0
            duration_1 = 100000
        else:
            duration_0 = 100000
            duration_1 = 0
    #

    # print('How many changes in dominance? ', np.sum([np.abs(flanks) == 1]))

    idx_start = np.where([flanks*1 == -1])[1]
    idx_end = np.where([flanks*1 == 1])[1]
    idx = np.sort(np.concatenate([idx_start, idx_end]))

    if visualization == 1:  # visualization
        # fig
        plt.plot(dominance_bool, 'r')
        plt.plot(summation2)
        plt.plot(idx, np.zeros(np.shape(idx)), 'ko')
        plt.draw()
        plt.savefig(file_base + '_dominance_periods.png')
        plt.close()
        # plt.show()

    return duration_0, duration_1  # dominance_duration_2, dominance_duration_1
#


def plot_dominance_durations_gerion(data_path, x_label, figure_name):
    # It is super ugly how I did this in the end. Really have to make this pretty at some point! 
    data_paths = glob(path.join(data_path, '*.pkl'))
    # file_path = data_paths[0]
    plot_data = dict()
    for file_path in data_paths:
        print(file_path)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            #

            if file_path == 'simulation_ext_attention_drive_[0.065,-0.065].pkl':
                print('break')

            file_base = path.splitext(file_path)[0]
            duration_0, duration_1 = dominance_duration(timecourse=data, dt=1, visualization=1,
                                                        file_base=file_base)

            # [dominance_duration_1.mean(), dominance_duration_2.mean()]
            attention_difference = np.diff([np.float(i) for i in file_base.split('[')[1].split(']')[0].split(',')])
            if attention_difference == 0:
                attention_difference = 0
            plot_data['%0.3f' % -attention_difference] = [np.mean(duration_0), np.mean(duration_1)]
        except Exception as e:
            print(e)
        #
    #

    cl = np.array([[0, 180, 77],
                   [230, 133, 36],
                   [230, 230, 36],
                   [75, 122, 191]
                   ]) / 255.

    lw = 2.5

    fig, ax = plt.subplots(1, 2, figsize=[10, 5])
    plt.rcParams.update({'font.size': 14})
    plot1_data = np.array([i[0] for i in plot_data.values()])
    plot2_data = np.array([i[1] for i in plot_data.values()])
    x = np.array([float(i) for i in plot_data.keys()])
    # ax[0].set_ylim([-0.001, 0.001])

    try:
        wta_ids = plot1_data + plot2_data > 10000
        wta_ids = np.where(wta_ids)[0][0]
        inf_line = plot1_data + plot2_data > 10000

        plot1_data[wta_ids:] = np.nan
        plot2_data[wta_ids:] = np.nan
    except:
        pass
    #

    ax[0].plot(x, 0.001 * np.array(plot1_data), color=cl[0], linewidth=lw, label='Orientation 1')
    ax[0].plot(x, 0.001 * np.array(plot2_data), color=cl[-1], linewidth=lw, label='Orientation 2')
    temp_ylim = ax[0].get_ylim()
    try:
        # inf_line = plot1_data + plot2_data > 10000
        inf_line = inf_line.astype(np.float)
        inf_id = np.where(inf_line)[0][0]
        pre_inf_id = inf_id-1
        inf_line = np.array(10000 * inf_line)
        inf_line[pre_inf_id] = 0.001 * np.max([plot1_data[pre_inf_id], plot2_data[pre_inf_id]])
        ax[0].plot(x[inf_line > 0], inf_line[inf_line > 0], color=cl[0], linestyle=(0, (5, 10)), linewidth=lw)
        ax[0].plot(x[inf_id:], 0 * inf_line[inf_id:], color=cl[-1], linewidth=lw)
        inf_line[inf_line >= 1000] = 0
        inf_line[pre_inf_id] = 0.001 * np.min([plot1_data[pre_inf_id], plot2_data[pre_inf_id]])
        ax[0].plot(x[inf_id-1:inf_id+1], inf_line[inf_id-1:inf_id+1], color=cl[-1], linestyle=(0, (5, 10)), linewidth=lw)
    except Exception as e:
        pass
    #

    try:
        # ax[0].set_ylim(temp_ylim)
        ax[0].legend()
        ratio = plot1_data[:pre_inf_id] / (plot1_data[:pre_inf_id] + plot2_data[:pre_inf_id])
        ax[1].plot(x[:pre_inf_id], 100 * ratio, color='k', linewidth=lw)
        ax[1].plot(x[pre_inf_id-1:pre_inf_id+1], [100 * ratio[-1], 100], color='k', linestyle=(0, (5, 10)), linewidth=lw)
        ax[1].plot(x[pre_inf_id:], 100 + 0 * x[pre_inf_id:], color='k', linewidth=lw)
        ax[0].set_ylim([-0.1, 6.4])
    except:
        ratio = plot1_data / (plot1_data + plot2_data)
        ax[1].plot(x, 100 * ratio, color='k', linewidth=lw)
    #

    ax[0].set_title('Summation dominance')
    ax[1].set_title('Relative dominance')

    ax[0].set_ylabel('dominance duration [sec]')
    ax[1].set_ylabel('relative dominance [%]')

    ax[0].set_xlabel(x_label)
    ax[1].set_xlabel(x_label)

    plt.tight_layout()
    plt.draw()
    plt.savefig(figure_name)

    plt.close()
#
