import numpy as np
from itertools import product
from dataclasses import dataclass
from typing import List
import pandas as pd
from network import Network

    
@dataclass(frozen=True)
class Snapshot:
    
    sensory_left_1: float
    sensory_left_2: float
    sensory_right_1: float
    sensory_right_2: float
    opponency_left_1: float
    opponency_left_2: float
    opponency_right_1: float
    opponency_right_2: float
    summation_1: float
    summation_2: float
    attention_1: float
    attention_2: float        
        
@dataclass(frozen=True)
class ExtendedSnapshot(Snapshot):
    sensory_left_1_excitation: float
    sensory_left_2_excitation: float
    sensory_right_1_excitation: float
    sensory_right_2_excitation: float
    summation_1_excitation: float
    summation_2_excitation: float
    opponency_left_1_excitation: float
    opponency_left_2_excitation: float
    opponency_right_1_excitation: float
    opponency_right_2_excitation: float
    attention_1_excitation: float
    attention_2_excitation: float
    sensory_left_1_suppression: float
    sensory_left_2_suppression: float
    sensory_right_1_suppression: float
    sensory_right_2_suppression: float
    summation_1_suppression: float
    summation_2_suppression: float
    opponency_left_1_suppression: float
    opponency_left_2_suppression: float
    opponency_right_1_suppression: float
    opponency_right_2_suppression: float
    attention_1_suppression: float
    attention_2_suppression: float
    sensory_left_1_habituation: float
    sensory_left_2_habituation: float
    sensory_right_1_habituation: float
    sensory_right_2_habituation: float
    summation_1_habituation: float
    summation_2_habituation: float
        
        
@dataclass
class Timecourse:
    
    snapshots: List[Snapshot]
        
    def append(self, snap):
        self.snapshots.append(snap)
        
    def get_neuron_over_time(self, neuron):
        return np.array([getattr(x, neuron) for x in self.snapshots])
        
        
def smooth_rectification(x, threshold=.05, slope=30):
    if x < 0:
        x = 0
    return x / (1 + np.exp( - slope * (x - threshold)))    


def non_smooth_rectification(x, n=1):
    if x < 0:
        x = 0
    return x ** n


class BaseClass:
    
    eyes = ['left', 'right']
    orientations = [1, 2]
    
    def __init__(
        self,
        smooth=True
    ):
        if smooth_rectification:
            self.rectification = smooth_rectification
        else:
            self.rectification = non_smooth_rectfication
            
            
def timecourse2pandas(timecourse):
    fields = timecourse.snapshots[-1].__dataclass_fields__.keys()
    return pd.DataFrame({
        field: timecourse.get_neuron_over_time(field) for field in fields
    })


#dictionary of parameters that define starting position for figure 2


def add_zero_startpoint(params):
    """
    update the params so that everything starts at 0
    and sensory_left_1 and opponency_left_1 start at fixed points
    """
    params[0].update({
        'init_response': {
            'left_1': 0.083404400940515,
            'left_2': 0.,
            'right_1': 0.,
            'right_2': 0.
        },
        'init_habituation': {
            key: 0. for key in ['left_1', 'left_2', 'right_1', 'right_2']
        }
    })
    params[2].update({
        'init_response': {
            'left_1': 0.144064898688432,
            'left_2': 0.,
            'right_1': 0.,
            'right_2': 0.
        },
        'init_habituation': {
            key: 0. for key in ['left_1', 'left_2', 'right_1', 'right_2']
        }
    })
    for lay in [1,3]:
        params[lay].update({
            'init_response': {
                key: 0. for key in ['1', '2']
            },
            'init_habituation': {
                key: 0. for key in ['1', '2']
            }
        })
    return params
    

def get_model_output(network_params, input_params, dt=.5):
    """
    generic function to run model given parameters and return data frame
    """
    network = Network(dt, *network_params)
    sensory_input = get_input(**input_params)
    timecourse = network.simulate(sensory_input)
    df = timecourse2pandas(timecourse)
    return df