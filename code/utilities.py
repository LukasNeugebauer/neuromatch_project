import numpy as np
from itertools import product
from dataclasses import dataclass
from typing import List

    
@dataclass
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
    pass
        
        
def smooth_rectification(x, threshold=.05, slope=30):
    x[x < 0] = 0.
    return x / (1 + np.exp( - slope * (x - threshold)))    


def non_smooth_rectification(x, n=1):
    x[x < 0] = 0.
    return x ** n


class BaseClass:
    
    eyes = ['left', 'right']
    orientations = [1, 2]
    
    def __init__(
        self,
        smooth_rectification=True
    ):
        if smooth_rectficication:
            self.rectification = smooth_rectification
        elif:
            self.rectification = non_smooth_rectfication