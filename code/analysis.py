"""
@author: kristinkaduk
"""

import numpy as np


def dominance_durations(timecourse, params):
    """
    average duration of all of the individual periods for which one of the rivalry stimuli dominates
    """

    time_vector = params[0]['time_vector']
    summation_1 = timecourse.get_neuron_over_time('summation_1')
    summation_2 = timecourse.get_neuron_over_time('summation_2')

    # timepoints at which summation neuron 2 is more active than 1
    dominance_2 = (summation_2 > summation_1).astype(np.int)
    #number of changes beween periods of dominance
    n_changes = (np.diff(dominance_2) != 0).sum()

    all_idx_start_1 = np.where(np.diff(dominance_2) == -1)[0]
    all_idx_start_2 = np.where(np.diff(dominance_2) == 1)[0]
    
    #only count full periods
    idx_start_dominance_1 = all_idx_start_1[all_idx_start_1 < all_idx_start_2.max()]
    idx_start_dominance_2 = all_idx_start_2[all_idx_start_2 < all_idx_start_1.max()]
    idx_end_dominance_1 = all_idx_start_2[all_idx_start_2 > all_idx_start_1.min()]
    idx_end_dominance_2 = all_idx_start_1[all_idx_start_1 > all_idx_start_2.min()]
    
    #average duration for orientation dominations
    duration_1 = time_vector[idx_end_dominance_1] - time_vector[idx_start_dominance_1]
    duration_2 = time_vector[idx_end_dominance_2] - time_vector[idx_start_dominance_2]
    
    return duration_1, duration_2


def ratio_dominance(timecourse, dominant=1):
    """
    return the relative dominance duration during timecourse.
    dominant = 1 -> summation_1 / (summation_1 + summation_2)
    dominant = 2 -> summation_2 / (summation_1 + summation_2)
    """
    dominant_summation = timecourse.get_neuron_over_time(f'summation_{int(dominant)}')
    other_summation = timecourse.get_neuron_over_time(f'summation_{int(3 - dominant)}')
    return (dominant_summation > other_summation).mean()