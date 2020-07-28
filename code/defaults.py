import numpy as np
import pandas as pd


def get_default_parameters(attended=True):
    dt = .5
    total_duration = 15000
    time_vector = np.arange(0, total_duration, dt)
    params = {
        'dt': dt,          # time-step in forward Euler method (ms)
        'total_duration': total_duration,       # total duration to simulate (ms)
        'time_vector': time_vector, # time vector (ms)
        'n_timepoints': time_vector.size,  # number of time point
        'n_locations': 1,           # simulate one location
        'n_orientations': 2,           # simulate two orienation
        'n': 2,  # exponent
        'sigma': .5,    # suppression constant (for all layer, excpet attention layer, see p.sigma_a below)
        'nLayers': 6,  # 2 monocular + 1 binocular-summation + 2 opponency + 1 attention layer
        'tau_habituation': 2000,  # time constant for adaptation
        'weight_habituation': 2,     # weights of self-adaptation
        'weight_opponency': .65,   # weights of mutual inhibition
        'weight_attention': .6,    # weights of attentional modulation
        'smooth_rectification': True
    }
    sensory_params, summation_params, opponency_params, attention_params = [params.copy() for _ in range(4)]
    sensory_params.update({
        'n': 1,
        'alpha': 2,     # gain (scaling factor) of monocualr neurons
        'tau_response': 5,     # time constant for monocular and binocualr-summation neurons
        'weight_attention': .6 if attended else 0
    })
    summation_params.update({
        'tau_response': 5
    })
    opponency_params.update({
        'tau_response': 20    # time constant for opponency neurons
    })
    attention_params.update({
        'sigma':  .2,              # suppression constant for attention layer
        'tau_response': 150   # time constant for attention
    })
    return sensory_params, summation_params, opponency_params, attention_params 
    
    
def get_input(dt, total_duration, tau=3, contrast=np.array([[.5, 0.], [0., .5]]), flicker=0, alpha_amp=.5):
    # todo: implement different
    # todo: onset modulation
    # sensory_input = np.empty(shape=(n_trials, 2), dtype=np.float)
    time_vector = np.arange(0, total_duration, dt)
    n_trials = time_vector.size
    input_dummy = np.ones((n_trials, 2))
    modulator = make_modulator(time_vector, tau, alpha_amp)
    input_left = input_dummy * contrast[0, :] 
    input_right = input_dummy * contrast[1, :]
    sensory_input = pd.DataFrame({
        'left_1': input_left[:, 0],
        'left_2': input_left[:, 1],
        'right_1': input_right[:, 0],
        'right_2': input_right[:, 1]
    }, index=np.arange(time_vector.size)) * modulator
    return sensory_input


def make_modulator(time_vector, tau, alpha_amp=.5, bound=10e-4):
    alpha = time_vector / tau * np.exp(1 - time_vector / tau)
    alpha = alpha[np.logical_or(time_vector <= tau, alpha >= bound)]
    modulator = np.ones(time_vector.size)
    modulator[:alpha.size] += alpha_amp * alpha
    return modulator[:, np.newaxis]