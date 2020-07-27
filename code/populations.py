import numpy as np
from itertools import product
import sys
sys.path.append('.')
from neurons import *
from utilities import *

class SensoryPopulation(BaseClass):
    
    def __init__(
        self,
        alpha,
        sigma,
        n,
        weight_opponency,
        weight_attention,
        weight_habituation,
        tau_response,
        tau_habituation, 
        smooth_rectification,
        initial_response=None,
        initial_habituation=None
    ):
        for eye, orientation in product(self.eyes, self.orientations):
            key = '_'.join([eye, str(orientation)])
            if key in initial_response.keys():
                _initial_response = initial_reponse[key]
            else:
                _initial_response = np.random.rand()
            if key in initial_habituation.keys():
                _initial_habituation = initial_habituation[key]
            else:    
                _initial_habituation = np.random.rand()
            self.neurons[key] = SensoryNeuron(
                eye=eye,
                orientation=orientation,
                alpha=alpha,
                sigma=sigma,
                n=n,
                weight_opponency=weight_opponency,
                weight_attention=weight_attention,
                weight_habituation=weight_habituation,
                tau_response=tau_response,
                tau_habituation=tau_habituation,
                initial_response=initial_response,
                initial_habituation=initial_habituation,
                smooth_rectification=smooth_rectification
            )
            
    def compute_excitatory_drive(
        self,
        sensory_input,
        snap
    ):
        for neuron in neurons.values():
            neuron.compute_excitatory_drive(sensory_input, snap)
        
    def update_state(
        self,
        sensory_input,
        snap
    ):
        for neuron in neurons.values():
            neuron.update_state(self.suppressive_drive, dt)
        
    @property
    def suppressive_drive(self):
        return sum([neuron.excitatory_drive for neuron in self.neurons.values()])


class SummationPopulation(BaseClass):
    
    def __init__(
        self,
        sigma,
        n,
        weight_habituation,
        tau_response,
        tau_habituation, 
        initial_response=None,
        initial_habituation=None
    ):
        for orientation in self.orientations:
            key = str(orientation)
            if key in initial_response.keys():
                _initial_response = initial_reponse[key]
            else:
                _initial_response = np.random.rand()
            if key in initial_habituation.keys():
                _initial_habituation = initial_habituation[key]
            else:    
                _initial_habituation = np.random.rand()
            self.neurons[key] = SummationNeuron(
                orientation=orientation,
                sigma=sigma,
                n=n,
                weight_habituation=weight_habituation,
                tau_response=tau_response,
                tau_habituation=tau_habituation,
                initial_response=initial_response,
                initial_habituation=initial_habituation
            )
            
    def compute_excitatory_drive(
        self,
        snapshot
    ):
        for neuron in self.neurons.values():
            neuron.compute_excitatory_drive(snapshot)
            
    def update_state(
        self
    ):
        for orientation in self.orientations:
            self.neurons[orientation].update_state()
            
            
class OpponencyPopulation(BaseClass):
        
    def __init__(
        self,
        sigma,
        n,
        tau_response,
        smooth_rectification,
        initial_response=None,
    ):
        for eye, orientation in product(self.eyes, self.orientations):
            key = '_'.join([eye, str(orientation)])
            if key in initial_response.keys():
                _initial_response = initial_reponse[key]
            else:
                _initial_response = np.random.rand()
            self.neurons[key] = OpponencyNeuron(
                eye=eye,
                orientation=orientation,
                sigma=sigma,
                n=n,
                tau_response=tau_response,
                initial_response=initial_response,
                smooth_rectification=smooth_rectification
            )
            
    def compute_excitatory_drives(
        self,
        snapshot
    ):
        for neuron in self.neurons.values():
            neuron.compute_excitatory_drive(snapshot)
        
    def update_state(
        self,
        dt
    ):
        suppressive_drives = self.suppressive_drives
        for eye in self.eyes:
            suppressive_drive_eye = suppressive_drives[eye]
            for orientation in self.orientations:
                key = '_'.join([eye, orientation])
                self.neurons[orientation].update_state(suppressive_drive_eye, dt)
                
    @property
    def suppressive_drives(self):
        drives = {}
        for eye in self.eyes:
            drives[eye] = sum([
                neurons['_'.join([eye, orientation])].excitatory_drive for orientation in self.orientations])
        return drives
    

class AttentionPopulation(BaseClass):
    
    def __init__(
        self,
        sigma,
        n,
        tau_response,
        smooth_rectification,
        initial_response=None
    ):
        super().__init__(smooth_rectification)
        self.neurons = {}
        for orientation in self.orientations:
            key = str(orientation)
            if key in initial_response.keys():
                _initial_response = initial_reponse[key]
            else:
                print(f'WARNING: Taking random start value for attention neuron for orientation {key}')
                _initial_response = np.random.rand()
            self.neurons[key] = AttentionNeuron(
                orientation=orientation,
                sigma=sigma,
                n=n,
                tau_response=tau_response,
                initial_response=_initial_response
            )
            
    def compute_excitatory_drives(
        self,
        snapshot
    ):
        for neuron in self.neurons.values():
            neuron.compute_excitatory_drive(snapshot)
        
    def update_state(
        self,
        dt
    ):
        for neuron in self.neurons.values():
            neuron.update_state(self.suppressive_drives, dt)
                
    @property
    def suppressive_drives(self):
        return np.sum(
            [self.rectification(neuron.excitatory_drive) for neuron in self.neurons.values()]
        )