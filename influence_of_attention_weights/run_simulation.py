from os.path import join, exists
import sys
import numpy as np
import pickle
sys.path.append('../code')
from network import Network
from defaults import get_input, get_default_parameters
from utilities import add_zero_startpoint

def overwrite_weight_attention(net, mean=.5, diff=0):
    for key in (f'{eye}_1' for eye in net.eyes):
        net.populations['sensory'].neurons[key].weight_attention = mean - diff
    for key in (f'{eye}_2' for eye in net.eyes):
        net.populations['sensory'].neurons[key].weight_attention = mean + diff
    return net


mean = .5
diffs = np.arange(.02, .26, .1)
folder = 'timecourses'

if __name__ == '__main__':
    
    dt = .1
    total_duration = 30000
    params = add_zero_startpoint(get_default_parameters())
    for pop_params in params:
        pop_params['total_duration'] = total_duration
        pop_params['dt'] = dt
    sensory_input = get_input(dt, total_duration)
    
    for diff in diffs:
        name = f'timecourse_weight-attention_mean-{mean}_diff-{diffs}.pic'
        savename = join(folder, name)
        if exists(savename):
            continue
        timecourse = overwrite_weight_attention(
            Network(dt, *params), 
            diff=diff, 
            mean = mean
        ).simulate(
            sensory_input
        )
        with open('savename', 'wb') as f:
            pickle.dump(timecourse, f, -1)