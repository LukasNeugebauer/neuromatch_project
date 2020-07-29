function p = setStim(p)

switch p.cond
    case {1,2} %Binocular Rivalry
        modulator = ones([1 p.nt]);
        onset     = makealpha(p.dt,p.alpha_t*100,p.alpha_t,10e-4); %skip this line if ignore onset transient
        modulator(1:length(onset)) = modulator(1:length(onset))+onset*p.alphaAmp; %skip this line if ignore onset transient
        
        OriL = [1 0]'; %Orientation in left  eye: one on and one off
        OriR = [0 1]'; %Orientation in right eye: one off and one on
        p.stimL = kron(ones(1,p.nt), OriL) * p.input(1) .* repmat(modulator,2,1);
        p.stimR = kron(ones(1,p.nt), OriR) * p.input(2) .* repmat(modulator,2,1);
        
    case {3,4} %Monocular Plaid
        modulator = ones([1 p.nt]);
        onset     = makealpha(p.dt,p.alpha_t*100,p.alpha_t,10e-4); %skip this line if ignore onset transient
        modulator(1:length(onset)) = modulator(1:length(onset))+onset*p.alphaAmp; %skip this line if ignore onset transient
        
        OriL = [1 1]'; %Monocualr plaid in left eye
        OriR = [0 0]'; %No stimulus in right eye
        p.stimL = kron(ones(1,p.nt), OriL) * p.input(1) .* repmat(modulator,2,1);
        p.stimR = kron(ones(1,p.nt), OriR) * p.input(2) .* repmat(modulator,2,1);
        
    case {5,6} %Blank
        ts_L     = zeros(p.ntheta,p.nt); %preallocate stimulus for left eye
        ts_R     = zeros(p.ntheta,p.nt); %preallocate stimulus for right eye
        ts_state = mod(floor(p.tlist/p.ISP),2)+1; %time series of the state of the stimulus (1 or 2, determine which stimulus is in which eye)
        
        %draw onset transient
        onsetIdx  = abs([1 diff(ts_state)]); %time series indexing the time point where stimulus swap
        ts_alpha  = conv(onsetIdx,makealpha(p.dt,p.alpha_t*100,p.alpha_t,10e-4)); %convolve onsetIdx with the shape of onset transient
        
        %add blank before swap
        modulator = ones(p.ntheta,p.nt);           %preallocate modulator of the stimulus (onset- offset- transient, and blank)
        changeIdx = find(abs([0 diff(ts_state)])); %time point where stimulus swap
        nblank    = round(p.blank/p.dt);           %length of blank in unit of dt
        for i = 1:nblank
            modulator(:,changeIdx-i)= 0; %insert blank before swap
        end
        
        %draw offset transient
        offsetIdx = double(([1 diff(modulator(1,:))])==-1);   %time series indexing the time point of stimulus offset
        ts_offset = conv(offsetIdx,makeoffset(p.dt,p.tan_t)); %convolve onsetIdx with the shape of offset decay
        
        modulator = modulator + repmat(ts_alpha(1:p.nt),p.ntheta,1)*p.alphaAmp + repmat(ts_offset(1:p.nt),p.ntheta,1);
        state_1 = [1 0]'; %stimulus for state one: one orientation one-one orientation off
        state_2 = [0 1]'; %stimulus for state two: one orientation one-off orientation on
        ts_L(:,ts_state==1) = repmat(state_1,1,sum(ts_state==1));
        ts_L(:,ts_state==2) = repmat(state_2,1,sum(ts_state==2));
        ts_R(:,ts_state==1) = repmat(state_2,1,sum(ts_state==1));
        ts_R(:,ts_state==2) = repmat(state_1,1,sum(ts_state==2));
        
        p.stimL = ts_L .* modulator * p.input(1);
        p.stimR = ts_R .* modulator * p.input(2);
        
    case 7 %Flicker and swap
        ts_L = zeros(p.ntheta,p.nt); %preallocate stimulus for left eye
        ts_R = zeros(p.ntheta,p.nt); %preallocate stimulus for right eye
        state_1 = [1 0]'; %stimulus for state one: one orientation one-one orientation off
        state_2 = [0 1]'; %stimulus for state two: one orientation one-off orientation on
        
        %draw flicker
        nframe_on  = round(1000/p.fHz/p.dt/2); %number of dt per on-cycle in the flikcer
        nrep       = ceil(p.nt / (nframe_on*2));
        flickerIdx = kron(ones(1,nrep),[ones(1,nframe_on) zeros(1, nframe_on)]); %we use even duty cycle
        flickerIdx = flickerIdx(1:p.nt);
        
        %add onset transient and offset decay to the flicker
        alpha = makealpha(p.dt,p.alpha_t*100,p.alpha_t,10e-4);
        alpha = alpha(1:min(nframe_on,length(alpha)));
        onsetIdx  = double(([1 diff(flickerIdx)])==1);
        offsetIdx = double(([1 diff(flickerIdx)])==-1);
        ts_alpha  = conv(onsetIdx,alpha);
        ts_offset = conv(offsetIdx,makeoffset(p.dt,p.tan_t));
        flickerIdx = repmat(flickerIdx+ts_alpha(1:p.nt)*p.alphaAmp+ts_offset(1:p.nt), p.ntheta,1);
        
        swapCycle = p.fHz/p.sHz;
        ts_state  = mod(floor(((1:p.nt)-1)/(swapCycle*nframe_on*2)),2)+1;
        ts_L(:,ts_state==1) = repmat(state_1,1,sum(ts_state==1));
        ts_L(:,ts_state==2) = repmat(state_2,1,sum(ts_state==2));
        ts_R(:,ts_state==1) = repmat(state_2,1,sum(ts_state==1));
        ts_R(:,ts_state==2) = repmat(state_1,1,sum(ts_state==2));
        p.stimL = ts_L.*flickerIdx * p.input(1);
        p.stimR = ts_R.*flickerIdx * p.input(2);
end
    function alpha = makealpha(dt,T,tau,bound)
        if ~exist('bound','var')
            bound = 10e-4;
        end
        tlist = 0:dt:T;
        alpha = tlist./tau.*exp(1-tlist/tau);
        alpha(tlist > tau & alpha<bound) = [];
    end
    function offset = makeoffset(dt,duration)
        xtanh  = linspace(pi-1,-pi+0.4,(duration+2)/dt);
        offset = 1/2*tanh(xtanh)+1/2;
        offset = offset - min(offset);
        offset = offset / max(offset);
        offset = offset(2:end-1);
    end
end