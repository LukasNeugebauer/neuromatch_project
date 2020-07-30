import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle
from os import path


def dominance_duration(timecourse, dt=0.5, visualization=1, file_base=''):
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
            duration_1 = dom_periods[dom_period_values == 1].mean()
        except:
            duration_1 = np.nan
        #
        try:
            duration_0 = dom_periods[dom_period_values == 0].mean()
        except:
            duration_0 = np.nan
        #
    except:
        duration_1 = np.nan
        duration_0 = np.nan
    #

    print('How many changes in dominance? ')
    print(np.sum([np.abs(flanks) == 1]))

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

    # a = np.arange(np.shape(idx)[0])
    # idx_odd = [num for num in a if num % 2 == 1]
    # idx_even = [num for num in a if num % 2 == 0]
    #
    # dominance_duration_2 = dt * (idx[idx_odd] - idx[idx_even[:-1]])
    # dominance_duration_1 = dt * (idx[idx_even[1:]] - idx[idx_odd])
    # # PercentageDominance_SumNeuron2 = (np.sum(dominance_bool == 1)) / np.size(dominance_bool)

    return duration_0, duration_1  # dominance_duration_2, dominance_duration_1
#


if __name__ == '__main__':
    data_paths = glob('*.pkl')
    # file_path = data_paths[0]
    plot_data = dict()
    for file_path in data_paths:
        print(file_path)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            #

            file_base = path.splitext(file_path)[0]
            duration_0, duration_1 = dominance_duration(timecourse=data, dt=1, visualization=1,
                                                        file_base=file_base)
            # if np.any(dominance_duration_1 < 0):
            #     print('pause')
            # print(duration_0, duration_1, duration_0.mean(), duration_1.mean())

            # [dominance_duration_1.mean(), dominance_duration_2.mean()]
            attention_difference = np.diff([np.float(i) for i in file_base.split('[')[1].split(']')[0].split(',')])
            if attention_difference == 0:
                attention_difference = 0
            plot_data['%0.3f' % -attention_difference] = [duration_0.mean(), duration_1.mean()]
            # plot_data['%0.3f' % -attention_difference] = [duration_0[-1], duration_1[-1]]
        except Exception as e:
            print(e)
        #
    #

    cl = np.array([[0, 180, 77],
                   [230, 133, 36],
                   [230, 230, 36],
                   [75, 122, 191]
                   ]) / 255.

    fig, ax = plt.subplots(1, 3)
    plot1_data = np.array([i[0] for i in plot_data.values()])
    plot2_data = np.array([i[1] for i in plot_data.values()])
    x = np.array([float(i) for i in plot_data.keys()])
    ax[0].plot(x, 0.001 * plot1_data, color=cl[-1])
    ax[1].plot(x, 0.001 * plot2_data, color=cl[-1])
    ax[2].plot(x, 100 * plot1_data / (plot1_data + plot2_data), color=cl[-1])

    ax[0].set_title('Summation dominance - Orientation 1')
    ax[1].set_title('Summation dominance - Orientation 2')
    ax[2].set_title('Relative dominance - Orientation 1')

    ax[0].set_ylabel('dominance duration [sec]')
    ax[1].set_ylabel('dominance duration [sec]')
    ax[2].set_ylabel('relative dominance [%]')

    # ax[0].set_xlabel('attention-drive difference')
    ax[1].set_xlabel('attention-drive difference')
    # ax[2].set_xlabel('attention-drive difference')

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    # plt.tight_layout()
    plt.draw()
    plt.savefig('comparision.png')

    print('Done')
#


# dominance_duration(timecourse=data, dt=0.5, visualization=1)

# sum_1 = data.get_neuron_over_time('summation_1')
# sum_2 = data.get_neuron_over_time('summation_2')
# dom = sum_1 > sum_2
# dom_switch_ids = np.where(abs(np.diff(dom)))[0]
# try:
#     if dom_switch_ids[0] < 10:
#         dom_switch_ids = dom_switch_ids[1:]
#     dom_periods = np.diff(dom_switch_ids)
#     dom_period_values = 1*(dom[dom_switch_ids+1])
#
#     print(np.array([dom_switch_ids, dom_periods, dom_period_values]).T)
#     plt.plot(abs(np.diff(dom)))
# except Exception as e:
#     print(e)
#     pass
# #
