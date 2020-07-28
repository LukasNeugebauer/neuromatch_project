import numpy as np
from itertools import product
import sys
sys.path.append('.')
from utilities import *


class SensoryNeuron(BaseClass):

    def __init__(
        self,
        eye,
        orientation,
        alpha,
        sigma,
        n,
        weight_opponency,
        weight_attention,
        weight_habituation,
        tau_response,
        tau_habituation,
        smooth_rectification,
        init_response,
        init_habituation,
        **kwargs
    ):
        super().__init__(smooth_rectification)
        self.eye = eye
        self.orientation = orientation
        self.alpha = alpha
        self.sigma = sigma
        self.n = n
        self.weight_opponency = weight_opponency
        self.weight_attention = weight_attention
        self.weight_habituation = weight_habituation
        self.tau_response = tau_response
        self.tau_habituation = tau_habituation
        self.response = init_response
        self.habituation = init_habituation
        self.opponency_response = 0

    def update_state(
        self,
        suppressive_drive,
        dt
    ):
        _change_in_habituation = self._calculate_change_in_habituation(dt)
        _change_in_response = self._calculate_change_in_response(suppressive_drive, dt)
        self.response += _change_in_response
        self.habituation += _change_in_habituation

    @property
    def excitatory_drive(
        self
    ):
        return self._excitatory_drive

    def compute_excitatory_drive(
        self,
        sensory_input,
        snapshot
    ):
        attention_response = getattr(snapshot, f'attention_{self.orientation}')
        try:
            self._excitatory_drive = self.rectification(
                sensory_input ** self.n - self.weight_opponency * self.opponency_response
            ) * self.rectification(
                1 + self.weight_attention * attention_response
            )
        except Exception as e:
            print(2)
            # raise e
            # from ipdb import set_trace
            # set_trace()

    def update_opponency_response(
            self,
            snapshot):
        if self.eye == 'left':
            other_eye = 'right'
        else:
            other_eye = 'left'
        self.opponency_response = np.sum(
            [getattr(snapshot, 'opponency_{}_{}'.format(other_eye, orientation)) for orientation in self.orientations]
        )

    def _calculate_change_in_response(
        self,
        suppressive_drive,
        dt
    ):
        change_in_response = (
            - self.response + 
            (self.alpha * self.excitatory_drive) / 
            (suppressive_drive + self.habituation ** self.n + self.sigma ** self.n)
        ) * (dt / self.tau_response)
        return change_in_response

    def _calculate_change_in_habituation(
        self,
        dt
    ):
        change_in_habituation = (
            - self.habituation +
            self.weight_habituation * self.response
        ) * (dt / self.tau_habituation)
        return change_in_habituation


class SummationNeuron(BaseClass):

    def __init__(
        self,
        orientation,
        sigma,
        n,
        weight_habituation,
        tau_response,
        tau_habituation,
        init_response,
        init_habituation,
        **kwargs
    ):
        self.orientation = orientation
        self.sigma = sigma
        self.n = n
        self.weight_habituation = weight_habituation
        self.tau_response = tau_response
        self.tau_habituation = tau_habituation
        self.response = init_response
        self.habituation = init_habituation

    def update_state(
        self,
        dt
    ):
        _change_in_habituation = self._calculate_change_in_habituation(dt)
        _change_in_response = self._calculate_change_in_response(dt)
        self.response += _change_in_response
        self.habituation += _change_in_habituation

    @property
    def excitatory_drive(
        self
    ):
        return self._excitatory_drive

    def compute_excitatory_drive(
        self,
        snapshot
    ):
        response_left = getattr(snapshot, f'sensory_left_{self.orientation}')
        response_right = getattr(snapshot, f'sensory_right_{self.orientation}')
        self._excitatory_drive = (response_left + response_right) ** self.n

    @property
    def suppressive_drive(
        self
    ):
        return self._excitatory_drive

    def _calculate_change_in_response(
        self,
        dt
    ):
        change_in_response = (
            - self.response + self.excitatory_drive / 
            (self.suppressive_drive + self.habituation ** self.n + self.sigma ** self.n)
        ) * (dt / self.tau_response)
        return change_in_response

    def _calculate_change_in_habituation(
        self,
        dt
    ):
        change_in_habituation = (
            - self.habituation +
            self.weight_habituation * self.response
        ) * (dt / self.tau_habituation)
        return change_in_habituation


class OpponencyNeuron(BaseClass):

    def __init__(
        self,
        eye,
        orientation,
        sigma,
        n,
        tau_response,
        init_response,
        smooth_rectification,
        **kwargs
    ):
        super().__init__(smooth_rectification)
        self.eye = eye
        self.orientation = orientation
        self.sigma = sigma
        self.n = n
        self.tau_response = tau_response
        self.response = init_response

    def update_state(
        self,
        suppressive_drive,
        dt
    ):
        _change_in_response = self.calculate_change_in_response(suppressive_drive, dt)
        self.response += _change_in_response

    @property
    def excitatory_drive(
        self
    ):
        return self._excitatory_drive

    def compute_excitatory_drive(
        self,
        snapshot
    ):
        other_eye = 'right' if self.eye == 'left' else 'left'
        response_same_eye = getattr(snapshot, f'sensory_{self.eye}_{self.orientation}')
        response_other_eye = getattr(snapshot, f'sensory_{other_eye}_{self.orientation}')
        self._excitatory_drive = self.rectification(response_same_eye - response_other_eye) ** self.n

    def calculate_change_in_response(
        self,
        suppressive_drive,
        dt
    ):
        change_in_response = (
            - self.response +
            self.excitatory_drive /
            (suppressive_drive + self.sigma ** self.n)
        ) * (dt / self.tau_response)
        return change_in_response


class AttentionNeuron(BaseClass):

    def __init__(
        self,
        orientation,
        sigma,
        n,
        tau_response,
        init_response,
        **kwargs
    ):
        self.orientation = orientation
        self.sigma = sigma
        self.n = n
        self.tau_response = tau_response
        self.response = init_response

    def compute_excitatory_drive(
        self,
        snapshot
    ):
        response_same_orientation = getattr(
            snapshot, f'summation_{self.orientation}'
        )
        response_other_orientation = getattr(
            snapshot, f'summation_{sum(self.orientations) - self.orientation}'
        )
        self._excitatory_drive = (response_same_orientation - response_other_orientation) ** self.n

    def update_state(
        self,
        suppressive_drive,
        dt
    ):
        _change_in_response = self.calculate_change_in_response(suppressive_drive, dt)
        self.response += _change_in_response

    @property
    def excitatory_drive(
        self
    ):
        return self._excitatory_drive

    def calculate_change_in_response(
        self,
        suppressive_drive,
        dt
    ):
        change_in_response = (
            - self.response +
            self.excitatory_drive /
            (suppressive_drive + self.sigma ** self.n)
        ) * (dt / self.tau_response)
        return change_in_response
