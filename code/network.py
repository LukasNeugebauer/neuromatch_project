import sys
sys.path.append('.')
from populations import *
from utilities import *
from tqdm import tqdm


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
        self.populations = None
        self.init_populations(
            sensory_population_arguments,
            summation_population_arguments,
            opponency_population_arguments,
            attention_population_arguments
        )
        
    @property
    def snapshot(self):
        return ExtendedSnapshot(
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
            attention_2 = self.populations['attention'].neurons['2'].response,
            sensory_left_1_excitation = self.populations['sensory'].neurons['left_1'].excitatory_drive,
            sensory_left_2_excitation = self.populations['sensory'].neurons['left_2'].excitatory_drive,
            sensory_right_1_excitation = self.populations['sensory'].neurons['right_1'].excitatory_drive,
            sensory_right_2_excitation = self.populations['sensory'].neurons['right_2'].excitatory_drive,
            summation_1_excitation = self.populations['summation'].neurons['1'].excitatory_drive,
            summation_2_excitation = self.populations['summation'].neurons['2'].excitatory_drive,
            opponency_left_1_excitation = self.populations['opponency'].neurons['left_1'].excitatory_drive,
            opponency_left_2_excitation = self.populations['opponency'].neurons['left_2'].excitatory_drive,
            opponency_right_1_excitation = self.populations['opponency'].neurons['right_1'].excitatory_drive,
            opponency_right_2_excitation = self.populations['opponency'].neurons['right_2'].excitatory_drive,
            attention_1_excitation = self.populations['attention'].neurons['1'].excitatory_drive,
            attention_2_excitation = self.populations['attention'].neurons['2'].excitatory_drive,
            sensory_left_1_suppression = self.populations['sensory'].suppressive_drive,
            sensory_left_2_suppression = self.populations['sensory'].suppressive_drive,
            sensory_right_1_suppression = self.populations['sensory'].suppressive_drive,
            sensory_right_2_suppression = self.populations['sensory'].suppressive_drive,
            summation_1_suppression = self.populations['summation'].neurons['1'].suppressive_drive,
            summation_2_suppression = self.populations['summation'].neurons['2'].suppressive_drive,
            opponency_left_1_suppression = self.populations['opponency'].suppressive_drives['left'],
            opponency_left_2_suppression = self.populations['opponency'].suppressive_drives['left'],
            opponency_right_1_suppression = self.populations['opponency'].suppressive_drives['right'],
            opponency_right_2_suppression = self.populations['opponency'].suppressive_drives['right'],
            attention_1_suppression = self.populations['attention'].suppressive_drives,
            attention_2_suppression = self.populations['attention'].suppressive_drives,
            sensory_left_1_habituation = self.populations['sensory'].neurons['left_1'].habituation,
            sensory_left_2_habituation = self.populations['sensory'].neurons['left_2'].habituation,
            sensory_right_1_habituation = self.populations['sensory'].neurons['right_1'].habituation,
            sensory_right_2_habituation = self.populations['sensory'].neurons['right_2'].habituation,
            summation_1_habituation = self.populations['summation'].neurons['1'].habituation,
            summation_2_habituation = self.populations['summation'].neurons['2'].habituation
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
        n_iter = sensory_input.shape[0]
        print(f'Running simulation for {n_iter} iterations.')
        for t in tqdm(range(n_iter)):
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

        # now update opponency_response
        for kind, population in self.populations.items():
            if kind == 'sensory':
                population.update_opponency_response(self.snapshot)

        for population in self.populations.values():
            population.update_state(self.dt)
            

    
def get_model_output(network_params, input_params, dt=.5):
    """
    generic function to run model given parameters and return data frame
    """
    network = Network(dt, *network_params)
    sensory_input = get_input(**input_params)
    timecourse = network.simulate(sensory_input)
    df = timecourse2pandas(timecourse)
    return df
    

def get_model_output(network_params, input_params, dt=.5):
    """
    generic function to run model given parameters and return data frame
    """
    network = Network(dt, *network_params)
    sensory_input = get_input(**input_params)
    timecourse = network.simulate(sensory_input)
    df = timecourse2pandas(timecourse)
    return df

def get_model_output(network_params, input_params, dt=.5):
    """
    generic function to run model given parameters and return data frame
    """
    network = Network(dt, *network_params)
    sensory_input = get_input(**input_params)
    timecourse = network.simulate(sensory_input)
    df = timecourse2pandas(timecourse)
    return df