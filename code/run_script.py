import numpy as np
import matplotlib.pyplot as plt
from network import Network
from defaults import get_input, get_default_parameters
import pickle


if __name__ == '__main__':
    params = get_default_parameters()

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
    params[2]['init_habituation'] = {'left_1': 0, 'left_2': 0, 'right_1': 0, 'right_2': 0}
    # attention population
    params[3]['init_response'] = {'1': 0, '2': 0}
    params[3]['init_habituation'] = {'1': 0, '2': 0}

    #
    network = Network(.5, *params)

    print(network.snapshot)

    dt = params[0]['dt']
    total_duration = params[0]['total_duration']

    sensory_input = get_input(dt, total_duration)
    # sensory_input = sensory_input[:30000]
    # plt.plot(sensory_input.values[:100])

    timecourse = network.simulate(sensory_input)
    summation_1 = timecourse.get_neuron_over_time('summation_1')
    summation_2 = timecourse.get_neuron_over_time('summation_2')

    print(summation_1)
    snap = timecourse.snapshots[-1]
    fields = snap.__dict__.keys()
    n_neurons = len(fields)

    fig, axs = plt.subplots(figsize=(30, n_neurons * 3), ncols=2, nrows=int(n_neurons / 2))
    for ax, field in zip(axs.flat, fields):
        ax.plot(timecourse.get_neuron_over_time(field))
        ax.set_title(field)

    plt.show()

    # save the results
    save_path = open('last_timecourse.pkl', 'wb')
    pickle.dump(timecourse, save_path)
    save_path.close()

    a = np.array([1.])
    print(a[a < 2])
#
