import sys
sys.path.append('.')
from populations import *
from utilities import *

class Network(BaseClass):
    
    def __init__(
        self, 
        dt,
        sensory_population_arguments,
        summation_population_arguments,
        opponency_population_arguments,
        attention_population_arguments,
    ):
        self.dt = dt
        self.init_populations(
            sensory_population_arguments,
            summation_population_arguments,
            opponency_population_arguments,
            attention_population_arguments
        )
        
    @property
    def snapshot(self):
        return Snapshot(
            sensory_left_1 = self.populations['sensory'].neurons['left_1'].response,
            sensory_left_2 = self.populations['sensory'].neurons['left_2'].response,
            sensory_right_1 = self.populations['sensory'].neurons['right_1'].response,
            sensory_right_2 = self.populations['sensory'].neurons['right_2'].response,
            summation_1 = self.populations['summation'].neurons['1'].response,
            summation_2 = self.populations['summation'].neurons['2'].response,
            opponency_left_1 = self.populations['opponency'].neurons['left_1'].response,
            opponency_left_2 = self.populations['opponency'].neurons['left_2'].response,
            opponency_right_1 = self.populations['opponency'].neurons['right_1'].response,
            opponency_right_2 = self.populations['opponency'].neurons['right_2'].response,
            attention_1 = self.populations['attention'].neurons['1'].response,
            attention_2 = self.populations['attention'].neurons['2'].response
        )
        
    def init_populations(
        self, 
        sensory_population_arguments,
        summation_population_arguments,
        opponency_population_arguments,
        attention_population_arguments
    ):
        self.populations = {}
        self.populations['sensory'] = SensoryPopulation(**sensory_population_arguments)
        self.populations['summation'] = SummationPopulation(**summation_population_arguments)
        self.populations['opponency'] = OpponencyPopulation(**opponency_population_arguments)
        self.populations['attention'] = AttentionPopulation(**attention_population_arguments)
        
    def simulate(self, sensory_input):
        """
        Expects n_timepoints x 2 sensory_input
        """
        timecourse = Timecourse([self.snapshot])
        for t in range(sensory_input.shape[0]):
            print(t, sensory_input.shape[0])
            if t == sensory_input.shape[0] - 1:
                self.one_step(sensory_input.iloc[t, :])
            else:
                self.one_step(sensory_input.iloc[t + 1, :])
            timecourse.append(self.snapshot)
        return timecourse
            
    def one_step(self, sensory_input):
        for kind, population in self.populations.items():
            if kind == 'sensory':
                population.compute_excitatory_drive(sensory_input, self.snapshot)
            else:
                population.compute_excitatory_drive(self.snapshot)
        for population in self.populations.values():
            population.update_state(self.dt)
            