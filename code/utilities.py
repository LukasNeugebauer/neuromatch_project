import numpy as np
from itertools import product
from dataclasses import dataclass
from typing import List

    
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