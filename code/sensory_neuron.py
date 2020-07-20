import numpy as np


def rectify(value):
	# we only want positive values. Negative values are set to 0.
	if value < 0:
		value = 0
	#
	return value
#

class sensory_neuron():
  def __init__(self, parameters):
    # parameters
    self.alpha = parameters['alpha']  # determins maximum response
    self.weight_o = parameters['weight_o']  # opponent weight
    self.weight_a = parameters['weight_a']  # attention weight
    self.sigma = parameters['sigma']
    self.n = parameters['n']
    self.weight_h = parameters['weight_h']  # adaptation weight
    self.tau_h = parameters['tau_h']
    self.tau_r = parameters['tau_r']

    # initialize state
    self.response = 0.5
    self.adaptation = 0.5
    # self.inputs = input_nodes
  #

  def calc_response(self, input_corresponding_eye, input_opposing_eye, opposing_response, attention_response):
    # calc excitatory_drive
    excitatory_drive = self._calc_excitatory_drive(input_corresponding_eye, opposing_response, attention_response)

    # calc suppressive_drive
    suppressive_drive = self._calc_suppressive_drive(input_corresponding_eye, input_opposing_eye)

    # update adaptation
    self._calc_adaptation(input_corresponding_eye)

    print(excitatory_drive, suppressive_drive, self.adaptation)

    print(self.alpha * excitatory_drive)
    print(suppressive_drive + self.adaptation + self.sigma**self.n)

    # calc the tau * d/dt * R
    change_in_response = - self.response + (self.alpha * excitatory_drive) / (suppressive_drive + self.adaptation + self.sigma**self.n)
    self.response += change_in_response / self.tau_r

    return self.response
  #
  
  def _calc_excitatory_drive(self, input_corresponding_eye, opposing_response, attention_response):
    input_rectified = rectify(np.sum(input_corresponding_eye[0] - self.weight_o * opposing_response))
    attention_rectified = rectify(np.sum(1 + self.weight_a * attention_response))
    
    # print(input_rectified, attention_rectified)

    return input_rectified * attention_rectified
  #

  def _calc_suppressive_drive(self, input_corresponding_eye, input_opposing_eye):
    return sum(input_corresponding_eye) + sum(input_opposing_eye)
  #

  def _calc_adaptation(self, input_corresponding_eye):
    drive = - self.adaptation + self.weight_h * input_corresponding_eye[0]
    self.adaptation += drive / self.tau_h
  #
#

# test the class
default_parameters = {
  'alpha': 1,  # determins maximum response
  'weight_o': 1,  # opponent weight
  'weight_a': 1,  # attention weight
  'weight_h': 1,  # adaptation weight
  'sigma': 1,
  'n': 1,
  'tau_r': 1,
  'tau_h': 1}

neuron1 = sensory_neuron(default_parameters)

response = neuron1.calc_response(input_corresponding_eye=[0.5], input_opposing_eye=[0.5], opposing_response=0.5, attention_response=0.5)
print(response)

