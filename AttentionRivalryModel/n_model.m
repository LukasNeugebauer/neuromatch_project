function [p] = n_model(p)

%Layers are defined as:
%Layer 1 = Left-eye monocular neurons
%Layer 2 = Right-eye monocular neurons
%Layer 3 = Binocular-summation neurons
%Layer 4 = Left-minus-right opponency neurons
%Layer 5 = Right-minus-left opponency neurons
%Layer 6 = Attention

% Define the rectification functions
switch p.rectSmoothFlag
    case 0
        h = @(x)halfExp(x,1);
    case 1
        h = @(x)halfExp_smooth(x);
end

idx = 1;
for t = p.dt:p.dt:p.T
    
%     if idx > 1
%         disp(idx)
%         disp(snapshot(p, idx-1))
%     end
    idx = idx+1;
%     [p.r{1}(:,1), p.r{2}(:,1), p.r{3}(:,1), p.r{4}(:,1), p.r{5}(:,1), p.r{6}(:,1)]

    % Print progress (msec) in command window
    if mod(p.tlist(idx),5000) <= p.dt/2
        counterdisp(p.tlist(idx));
    end
    
    %% Monocular Layers
    for lay = [1 2]
        % Input: for monocualr layer the input is the sitmulus input strength
        inp = p.i{lay}(:,idx);
        
        % Updating excitatory drive: E
        p.d{lay}(:,idx) =h(inp.^p.n_m - p.o{lay}(:,idx-1)*p.wo) .* h(1 + p.r{6}(:,idx-1)*p.wa);
    end
    for lay = [1 2]
        % Defining normalization pool (four monocualr neurons)
        pool = [p.d{1}(:,idx) p.d{2}(:,idx)];
        
        % Compute suppressive drive: S
        p.s{lay}(:,idx) = sum(pool(:)); % suppressive drive: sum over all the units in normalization pool
        
        % Asymptotic firing rate (normalization equation)
        p.f{lay}(:,idx) = p.m*p.d{lay}(:,idx) ./ (p.s{lay}(:,idx) + p.sigma.^p.n_m + p.h{lay}(:,idx-1).^p.n_m);
        
        % Update response: R
        p.r{lay}(:,idx) = p.r{lay}(:,idx-1) + (p.dt/p.tau_s)*(-p.r{lay}(:,idx-1) + p.f{lay}(:,idx));
        
        % Update adaptation: H
        p.h{lay}(:,idx) = p.h{lay}(:,idx-1) + (p.dt/p.tau_h)*(-p.h{lay}(:,idx-1) + p.r{lay}(:,idx-1)*p.wh);
    end
    
    %% Binocular-summation and Opponency Layers
    for lay = 3:5
        % Input
        switch lay
            case 3 %Binocular-summation neurons
                inp = p.r{1}(:,idx-1) + p.r{2}(:,idx-1);
            case 4 %LE-RE opponency neurons
                inp = h(p.r{1}(:,idx-1) - p.r{2}(:,idx-1));
            case 5 %RE-LE opponency neurons
                inp = h(p.r{2}(:,idx-1) - p.r{1}(:,idx-1));
        end
        
        % Updating excitatory drive: E
        p.d{lay}(:,idx) = inp.^p.n;
    end
    for lay = 3:5
        % Compute suppressive drive (S), asymptotic response (normalization equation) (f), update response (R)
        switch lay
            case 3
                p.s{lay}(:,idx) = p.d{lay}(:,idx);
                p.f{lay}(:,idx) = p.d{lay}(:,idx) ./ (p.s{lay}(:,idx) + p.sigma.^p.n + p.h{lay}(:,idx-1).^p.n);
                p.r{lay}(:,idx) = p.r{lay}(:,idx-1) + (p.dt/p.tau_s)*(-p.r{lay}(:,idx-1) + p.f{lay}(:,idx));
            case {4,5}
                pool = p.d{lay}(:,idx);
                p.s{lay}(:,idx) = sum(pool(:));
                p.f{lay}(:,idx) = p.d{lay}(:,idx) ./ (p.s{lay}(:,idx) + p.sigma.^p.n);
                p.r{lay}(:,idx) = p.r{lay}(:,idx-1) + (p.dt/p.tau_o)*(-p.r{lay}(:,idx-1) + p.f{lay}(:,idx));
        end
        
        % Update adaptation: H
        if lay == 3
            p.h{lay}(:,idx) = p.h{lay}(:,idx-1) + (p.dt/p.tau_h)*(-p.h{lay}(:,idx-1) + p.r{lay}(:,idx)*p.wh);
        end
        
        % Pooling the responses of opponency neurons for each eye, for eliciting inhibition
        if lay == 4
            p.o{2}(:,idx) = sum(p.r{lay}(:,idx)); % Inhibition sent to lay 2
        elseif lay == 5
            p.o{1}(:,idx) = sum(p.r{lay}(:,idx)); % Inhibition sent to lay 1
        end
    end
    
    %% Update attention
    for lay=6
        % This is written in an alternative form of Eq.3 in the paper.
        % This is a short-hand for combining on- and off- channels,
        % aSign keeps track of the sign of attention (attentional enhacement vs. attentional suppression)
        % so this will work regardless whether the exponent term is an an odd or even number
        if 0
            % OVERWRITE GERION JUST FOR DEBUGGING
            inp     = [0.1; 0.3];
            aDrive  = abs(p.aKernel*inp);
            aSign   = sign(p.aKernel*inp);
        else
            inp     = p.r{3}(:,idx);
            aDrive  = abs(p.aKernel*inp);
            aSign   = sign(p.aKernel*inp);
        end
        
        % Excitatory drive
        p.d{lay}(:,idx) = aSign.*(aDrive.^p.n);
        
        % Suppressive drive (S) + suppression constant (sigma)
        p.s{lay}(:,idx) = repmat((sum(aDrive.^p.n) + p.sigma_a^p.n),p.ntheta,1);
        
        % Asymptotic firing rate (normalization equation)
        p.f{lay}(:,idx) = p.d{lay}(:,idx) ./ p.s{lay}(:,idx);
        
        % Update responses: R
        p.r{lay}(:,idx) = p.r{lay}(:,idx-1) + (p.dt/p.tau_a)*(-p.r{lay}(:,idx-1) + p.f{lay}(:,idx));
    end
end
    % Function for displaying progress
    function counterdisp(i)
        fprintf('%d msec \r', i);
    end

    % Rectification (non-smoothed version)
    function [x] = halfExp(base,n)
        if nargin == 1
            n=1;
        end
        x = (max(0,base)).^n;
    end

    % Rectification (smoothed version)
    function y=halfExp_smooth(x)
        thresh=0.05;
        slope=30;
        
        idx_0 = x<0;
        idx_1 = x>0;
        
        y = zeros(size(x));
        y(idx_0) = 0;
        y(idx_1) = x(idx_1).*1./(1+exp(-slope*(x(idx_1)-thresh)));
    end
end


function [snapshot] = snapshot(p, idx)
    snapshot = [];
    snapshot.attention_1 = p.r{6}(1, idx);
    snapshot.attention_2 = p.r{6}(2, idx);
    snapshot.opponency_left_1 = p.r{4}(1, idx);
    snapshot.opponency_left_2 = p.r{4}(2, idx);
    snapshot.opponency_right_1 = p.r{5}(1, idx);
    snapshot.opponency_right_2 = p.r{5}(2, idx);
    snapshot.sensory_left_1 = p.r{1}(1, idx);
    snapshot.sensory_left_2 = p.r{1}(2, idx);
    snapshot.sensory_right_1 = p.r{2}(1, idx);
    snapshot.sensory_right_2 = p.r{2}(2, idx);
    snapshot.summary_1 = p.r{3}(1, idx);
    snapshot.summary_2 = p.r{3}(2, idx);
end


