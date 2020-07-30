import sys
sys.path.append('.')
from modules.neurons import *
from modules.utilities import *


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
        init_response={},
        init_habituation={},
        **kwargs
    ):
        self.neurons = {}
        for eye, orientation in product(self.eyes, self.orientations):
            key = ''.join([eye, '_', str(orientation)])
            if key in init_response.keys():
                _init_response = init_response[key]
            else:
                print(f'WARNING: Taking random start value for sensory neuron for eye {eye}, orientation {orientation}')
                _init_response = np.random.rand()
            if key in init_habituation.keys():
                _init_habituation = init_habituation[key]
            else:    
                print(f'WARNING: Taking random start habituation value for sensory neuron for eye {eye}, orientation {orientation}')
                _init_habituation = np.random.rand()
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
                init_response=_init_response,
                init_habituation=_init_habituation,
                smooth_rectification=smooth_rectification
            )
            
    def compute_excitatory_drive(
        self,
        sensory_input,
        snap
    ):
        for key, neuron in self.neurons.items():
            neuron.compute_excitatory_drive(sensory_input[key], snap)
        
    def update_state(
        self,
        dt
    ):
        for neuron in self.neurons.values():
            neuron.update_state(self.suppressive_drive, dt)

    def update_opponency_response(
            self,
            snap
    ):
        for key, neuron in self.neurons.items():
            neuron.update_opponency_response(snap)

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
        init_response={},
        init_habituation={},
        **kwargs
    ):
        self.neurons = {}
        for orientation in self.orientations:
            key = str(orientation)
            if key in init_response.keys():
                _init_response = init_response[key]
            else:
                print(f'WARNING: Taking random start value for summation neuron for orientation {orientation}')
                _init_response = np.random.rand()
            if key in init_habituation.keys():
                _init_habituation = init_habituation[key]
            else:    
                print(f'WARNING: Taking random start habituation value for summation neuron for orientation {orientation}')
                _init_habituation = np.random.rand()
            self.neurons[key] = SummationNeuron(
                orientation=orientation,
                sigma=sigma,
                n=n,
                weight_habituation=weight_habituation,
                tau_response=tau_response,
                tau_habituation=tau_habituation,
                init_response=_init_response,
                init_habituation=_init_habituation
            )
            
    def compute_excitatory_drive(
        self,
        snapshot
    ):
        for neuron in self.neurons.values():
            neuron.compute_excitatory_drive(snapshot)
            
    def update_state(
        self,
        dt
    ):
        for orientation in self.orientations:
            self.neurons[str(orientation)].update_state(dt)
            
            
class OpponencyPopulation(BaseClass):
        
    def __init__(
        self,
        sigma,
        n,
        tau_response,
        smooth_rectification,
        init_response={},
        **kwargs
    ):
        self.neurons = {}
        for eye, orientation in product(self.eyes, self.orientations):
            key = ''.join([eye, '_', str(orientation)])
            if key in init_response.keys():
                _init_response = init_response[key]
            else:
                print(f'WARNING: Taking random start value for opponency neuron for eye {eye}, orientation {orientation}')
                _init_response = np.random.rand()
            self.neurons[key] = OpponencyNeuron(
                eye=eye,
                orientation=orientation,
                sigma=sigma,
                n=n,
                tau_response=tau_response,
                init_response=_init_response,
                smooth_rectification=smooth_rectification
            )
            
    def compute_excitatory_drive(
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
                key = ''.join([eye, '_', str(orientation)])
                self.neurons[key].update_state(suppressive_drive_eye, dt)
                
    @property
    def suppressive_drives(self):
        drives = {}
        for eye in self.eyes:
            drives[eye] = sum([
                self.neurons['_'.join([eye, str(orientation)])].excitatory_drive for orientation in self.orientations])
        return drives
    

class AttentionPopulation(BaseClass):
    
    def __init__(
        self,
        sigma,
        n,
        tau_response,
        smooth_rectification,
        init_response={},
        **kwargs
    ):
        super().__init__(smooth_rectification)
        self.neurons = {}
        for orientation in self.orientations:
            key = str(orientation)
            if key in init_response.keys():
                _init_response = init_response[key]
            else:
                print(f'WARNING: Taking random start value for attention neuron for orientation {key}')
                _init_response = np.random.rand()
            self.neurons[key] = AttentionNeuron(
                orientation=orientation,
                sigma=sigma,
                n=n,
                tau_response=tau_response,
                init_response=_init_response
            )
            
    def compute_excitatory_drive(
        self,
        snapshot,
        **kwagrs
    ):
        for neuron in self.neurons.values():
            neuron.compute_excitatory_drive(snapshot, **kwagrs)
        
    def update_state(
        self,
        dt
    ):
        for neuron in self.neurons.values():
            neuron.update_state(self.suppressive_drives, dt)
                
    @property
    def suppressive_drives(self):
        # [self.rectification(neuron.excitatory_drive) for neuron in self.neurons.values()]
        return np.sum(
            [np.abs(neuron.excitatory_drive) for neuron in self.neurons.values()]
        )