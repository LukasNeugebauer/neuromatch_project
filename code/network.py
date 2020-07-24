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
        self.init_populations(
            sensory_population_arguments,
            summation_population_arguments,
            opponency_population_arguments,
            attention_populations_arguments
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
        attention_population_arguments,
        opponency_population_arguments
    ):
        self.populations = {}
        self.populations['sensory'] = SensoryPopulation(**sensory_population_arguments)
        self.populations['summation'] = SummationPopulation(**summation_population_arguments)
        self.populations['attention'] = AttentionPopulation(**attention_population_arguments)
        self.populations['opponency'] = OpponencyPopulation(**opponency_populations_arguments)
        
    def simulate(self, sensory_input):
        """
        Expects n_timepoints x 2 sensory_input
        """
        timecourse = Timecourse([self.snapshot])
        for t in sensory_input.shape[0]:
            self.one_step(sensory_input[t, :])
            timecourse.append(self.snapshot)
        return timecourse
            
    def one_step(self, sensory_input):
        for population in self.populations:
            population.compute_excitatory_drive(self.snapshot)
        for population in self.populations:
            population.update_state()