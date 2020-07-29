import numpy as np
import matplotlib.pyplot as plt
from network import Network
from defaults import get_input, get_default_parameters
# import pickle
from unpack_mat_data import load_mat_data


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


if __name__ == '__main__':
    # get params and init to matlab seed
    params = get_default_parameters()
    init_state(params)

    # initialize the network
    network = Network(.5, *params)
    # print(network.snapshot)

    dt = params[0]['dt']
    total_duration = params[0]['total_duration']
    sensory_input = get_input(dt, total_duration)

    sensory_input = sensory_input[:30000]

    timecourse = network.simulate(sensory_input)

    n = 30000
    mat_data = load_mat_data(file_path=r'matlab_timecourse.mat')

    fig, ax = plt.subplots(nrows=6, ncols=2)
    all_keys = mat_data['r'].keys()
    i = 0
    for key in all_keys:
        ax_i = int(np.floor(i / 2))
        ax_j = np.mod(i, 2)
        ax[ax_i, ax_j].plot(np.array([mat_data['r'][key][0, :n], timecourse.get_neuron_over_time(key)[:n]]).T)
        ax[ax_i, ax_j].set_title(key)
        if 'attention' in key:
            ax[ax_i, ax_j].set_ylim([-0.5, 0.5])
        else:
            ax[ax_i, ax_j].set_ylim([0., 1.])
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

    # # save the results
    # save_path = open('last_timecourse.pkl', 'wb')
    # pickle.dump(timecourse, save_path)
    # save_path.close()
#
