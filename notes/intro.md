# Attention model of binocular rivalry

## Abstract

Competing image supresses rival through gain changes
 * Similar to attention
 * Link to attention

Two main components of the model:
 * Attention
 * Mutual inhibition

These have different characteristics:
 * Feature (attention) vs. eye (inhibition)
 * fast (inhibition) vs. slow (attention)

Hallmarks:
 * attention required
 * different states when images are swapped at high frequency (don't really understand what that's supposed to mean. have to go back later. That is, depending on the frequency of change the percept follows one eye or it follows one stimulus
 * Level't proposition (look up) predicts dominance duration as a function of input strength. 


## Intro

Classical models consider mainly lateral inhibition and are driven by bottom up input. But data shows that rivalry requires attention. 
New model: Stimuli compete for attention. Attention is modeled as multiplicative gain (have to look at note for this one). Is present when attention is on stimuli. Higher sensory activation leads to multiplicative gain and drives attention away from other stimulus.
Mutual inhibition is modeled by opponency neurons that take in input from monocular neurons and supress output for one.
	

## Methods

### Type of model

No single neuron model. "Neuron" output in model is supposed to be linked to mean firing rate of an ensemble of neurons with similar firing properties. Canonical input-output-function is assumed. Model is supposed to capture the signal-processing related activity in visual cortex, not the biophysical mechanism.

### General structure

3 classes of neurons responsible for:
 * sensory representation
 * attentional modulation
 * mutual inhibition

### Sensory representation

6 Total neurons for sensory representations, divided in to three populations with two neurons each. One each per eye and one for binocular summation:

 * Left eye neurons (LE)
 * Right eye neurons (RE)
 * Binocular summation neurons

There are two neurons per population, one for each of two orthogonal representations.

#### Monocular neurons

Four monocular neurons are similar, with different inputs. These neurons are characterized as *R[i,j]* where i and j are subscripts for eye and orientation.

Change in response (times a time constant) depends on:

 * E: Excitatory drive
 * S: Suppressive drive
 * H: Adaptation

Excitatory drive depends on stimulus property and is determined by input *D*.
E = [D^n - wo*O]+ [1 + wa*Ra]+
D depends on stimulus contrast and increases monotonically with increasing contrast.
O is the mutual inhibition from opponency neuron that respons to the opposite eye. I'm not yet sure how that works.
[1 + wa * Ra] is the attention gain factor. 1 is baseline attention gain and Ra1 is the activation of the respective attention neuron. Will have to look this up.

wo and wa are free parameters of the model and determine the influence of mutual inhibition and attention gain respectively.

[]+ means half-wave rectification. In practice it's something like the exponential of the non-negative input and 0 if input is negative.

S is the supressive drive, the sum of all excitatory drives. I.e. it's role is scaling.

n and \sigma determine dose-response relationship between the input and the excitatory drive

\alpha determines the max response.

\tau is a time constant, the unit of updating.

H is the adaptation term that slowly brings neurons back to baseline, depending on the parameter wh.

#### Summation neurons

Summation neurons work similar to monocular neurons. The excitatory drive is a power of the sum of the two monocular neurons per eye. Supression is the same as the excitatory drive. And there is also a self-adapting term that depends on wh. I assume wh is the same for all neurons. Binocular neurons do not show mutual inhibition in this model. Here they are used to prove input to the attention layer.

### Attention modulation

Stronger orientation gets stronger attention gain. There are two attention neurons in the attention layer. One neuron per orientation.  The excitatory drive of this neuron depends on the difference in activation of the summation neurons. This difference is raised to some power. 
The supressive drive is the sum of the half-rectified activation of the excitatory drive of both attention neurons. Somehow this accomodates the negative response of some neurons but I'm not sure how that works. Maybe this way the attention weight in total can not get below 0. 
The time constant of attentional modulation is considerably slower of the time frame of sensory input. This is probably due to theoretical constraints. 

### Mutual inhibition

Is mediated through opponency neurons. There are 4, one for each combination of eye and orientation. I.e. one pair of left-minus-right, one pair of right-minus-left neurons. 
The excitatory drive of the right-eye opponency-neuron selective for orientation _a_ depends on the difference of the activation of the right- and left-eye neurons for this orientation. Supressive drive was computed over orientations but not over eyes. I.e. S = sum(Eor1, Eor2). Both right-eye opponency neuron activations were summed up in **O**r (for the right eye) and act inhibitory on the excitatory drive of the left monocular neurons.
These opponency neurons don't inhibit the opposing eye if there's no input in their respective retina. I.e. with stationary monocular-plaid stimuli, they don't predict any changing percepts.



