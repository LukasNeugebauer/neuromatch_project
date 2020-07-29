import numpy as np
import matplotlib.pyplot as plt
from network import Network
from defaults import get_input, get_default_parameters
# import pickle
from unpack_mat_data import load_mat_data


def plot_activity(timecourse, dt):
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=[8, 10])
    all_keys = ['sensory activity',
                'opponency activity',
                'summation activity',
                'attention activity']

    # i = 0
    x = np.arange(0, timecourse.get_neuron_over_time('sensory_left_1').size) * dt / 1000
    for i in range(ax.size):
        key = all_keys[i]
        # ax_i = int(np.floor(i / 2))
        # ax_j = np.mod(i, 2)
        ax_i = i
        # ax_j = 0
        if i == 0:
            neuron_keys = ['sensory_left_1', 'sensory_left_2', 'sensory_right_1', 'sensory_right_2']
            labels = ['left orientation 1', 'left orientation 2', 'right orientation 1', 'right orientation 2']
        elif i == 1:
            # neuron_keys = ['opponency_left_1', 'opponency_left_2', 'opponency_right_1', 'opponency_right_2']
            # labels = ['left', 'right']
            ax[ax_i].plot(x, np.array([
                timecourse.get_neuron_over_time('opponency_left_1') +
                timecourse.get_neuron_over_time('opponency_left_2')
            ]).T, label='left')
            ax[ax_i].plot(x, np.array([
                timecourse.get_neuron_over_time('opponency_right_1') +
                timecourse.get_neuron_over_time('opponency_right_2')
            ]).T, label='right')
        elif i == 2:
            neuron_keys = ['summation_1', 'summation_2']
            labels = ['orientation 1', 'orientation 2']
        elif i == 3:
            neuron_keys = ['attention_1', 'attention_2']
            labels = ['orientation 1', 'orientation 2']
        else:
            ax[ax_i].plot(x, np.array([timecourse.get_neuron_over_time(key)]).T)
        #

        if i != 1:
            for neuron_id, label in zip(neuron_keys, labels):
                ax[ax_i].plot(x, timecourse.get_neuron_over_time(neuron_id).T, label=label)

        ax[ax_i].set_title(key)
        if 'attention' in key:
            ax[ax_i].set_ylim([-0.8, 0.8])
        else:
            ax[ax_i].set_ylim([-0.2, 1.2])
        #
        ax[ax_i].legend(loc=1)
        i += 1
    #
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    plt.tight_layout()
    plt.draw()
    plt.savefig('Activity_last_run.png')
    plt.show()
#


def comparision_python_matlab_plot(timecourse):
    n = 30000
    mat_data = load_mat_data(file_path=r'matlab_timecourses\matlab_timecourse_cond_1.mat')

    fig, ax = plt.subplots(nrows=6, ncols=2)
    all_keys = mat_data['r'].keys()
    i = 0
    for key in all_keys:
        ax_i = int(np.floor(i / 2))
        ax_j = np.mod(i, 2)
        ax[ax_i, ax_j].plot(np.array([mat_data['r'][key][0, :n], timecourse.get_neuron_over_time(key)[:n]]).T)
        ax[ax_i, ax_j].set_title(key)
        if 'attention' in key:
            ax[ax_i, ax_j].set_ylim([-0.8, 0.8])
        else:
            ax[ax_i, ax_j].set_ylim([-0.2, 1.2])
        #
        i += 1
    #
    # mng = plt.get_current_fig_manager()
    # mng.frame.Maximize(True)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.draw()
    plt.savefig('Comparrision Matlab and Python.png')
    plt.show()
#


def init_state(params):
    # sensory
    # sens_l_rand_matlab = np.random.rand() * 0.2
    sens_l_rand_matlab = 0.083404400940515
    params[0]['init_response'] = {'left_1': sens_l_rand_matlab, 'left_2': sens_l_rand_matlab, 'right_1': 0, 'right_2': 0}
    params[0]['init_habituation'] = {'left_1': 0, 'left_2': 0, 'right_1': 0, 'right_2': 0}
    # summation
    params[1]['init_response'] = {'1': 0, '2': 0}
    params[1]['init_habituation'] = {'1': 0, '2': 0}
    # opponency population
    opp_l_rand_matlab = 0.144064898688432
    # params[2]['init_response'] = {'left_1': 0, 'left_2': 0, 'right_1': 0, 'right_2': 0}
    params[2]['init_response'] = {'left_1': opp_l_rand_matlab, 'left_2': opp_l_rand_matlab, 'right_1': 0, 'right_2': 0}
    # params[2]['init_response'] = {'left_1': 0, 'left_2': 0, 'right_1': opp_l_rand_matlab, 'right_2': opp_l_rand_matlab}
    params[2]['init_habituation'] = {'left_1': 0, 'left_2': 0, 'right_1': 0, 'right_2': 0}
    # attention population
    params[3]['init_response'] = {'1': 0, '2': 0}
    params[3]['init_habituation'] = {'1': 0, '2': 0}

    return params
#


def overwrite_weight_attention(net, new_wa=[0.7, 0.5]):
    for key in net.populations['sensory'].neurons.keys():
        if '1' in key:
            net.populations['sensory'].neurons[key].weight_attention = new_wa[0]
        else:
            net.populations['sensory'].neurons[key].weight_attention = new_wa[1]
        #

    return net
#


if __name__ == '__main__':
    # get params and init to matlab seed
    params = get_default_parameters()
    init_state(params)

    # initialize the network
    network = Network(.5, *params)
    # print(network.snapshot)

    network = overwrite_weight_attention(network, new_wa=[0.9, 0.3])

    dt = params[0]['dt']
    total_duration = params[0]['total_duration']
    sensory_input = get_input(dt, total_duration)
    sensory_input = sensory_input[:30000]

    timecourse = network.simulate(sensory_input)

    # comparision_python_matlab_plot(timecourse)
    plot_activity(timecourse, dt)

    # # save the results
    # save_path = open('last_timecourse.pkl', 'wb')
    # pickle.dump(timecourse, save_path)
    # save_path.close()
#
