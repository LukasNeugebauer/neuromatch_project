

def get_default_parameters(attended=True):
    params = {
        'dt'     = .5;         %time-step in forward Euler method (ms)
        p.T      = 15000;      %total duration to simulate (ms)
        p.nt     = p.T/p.dt+1; %number of time point
        p.tlist  = 0:p.dt:p.T; %time vector (ms)
        p.nx     = 1;          %simulate one location
        p.ntheta = 2;          %simulate two orienation

        % some properties of simulated neurons
        p.nLayers       = 6; %2 monocular + 1 binocular-summation + 2 opponency + 1 attention layer
        p.rectSmoothFlag= 1; %options for half-wave rectification function. 0 is non-smoothed version; 1 is the smoothed version; both work here.
                             %if you want to implement the model using ODE solver in MATLAB (or in Mathematica), or to run the model in AUTOp07, 
                             %the smoothed version is preferred.

        % model parameters
        p.n_m   = 1;   %exponent for monocualr neurons
        p.n     = 2;   %exponent for all other neurons
        p.sigma = .5;  %suppression constant (for all layer, excpet attention layer, see p.sigma_a below)
        p.m     = 2;   %gain (scaling factor) of monocualr neurons
        p.wh    = 2;   %weights of self-adaptation
        p.wo    = .65; %weights of mutual inhibition
        p.wa    = .6;  %weights of attentional modulation
        p.tau_s = 5;   %time constant for monocular and binocualr-summation neurons
                       %The range 1-10 ms has been tested, and generated similar results
        p.tau_o = 20;  %time constant for opponency neurons
        p.tau_a = 150; %time constant for attention
        p.tau_h = 2000;%time constant for adaptation

        % some stuff for attention layer
        p.sigma_a = .2;          %suppression constant for attention layer
        p.aKernel = [1 -1;-1 1]; %weight from binocualr summation neurons to attention neurons (subtraction in Eq.3 in the paper)

    }
    if attended:
        params['weight_attention'] = .6
    else:
        params['weight_attention'] = 0
    return params
    
    
def get_input(n_cond, n_trials, contrast):
    pass