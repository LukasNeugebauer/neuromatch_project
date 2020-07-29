function p = initTimeSeries(p)

for lay=1:p.nLayers %go through maximum possible layers. This way, if there are <5 layers, the feedback can be zero.
    p.d{lay} = zeros(p.ntheta,p.nt); %Excitatory Drive
    p.s{lay} = zeros(p.ntheta,p.nt); %Suppressive Drive
    p.r{lay} = zeros(p.ntheta,p.nt); %Firing Rate
    p.f{lay} = zeros(p.ntheta,p.nt); %Estimated asymptotic firing rate
    p.h{lay} = zeros(p.ntheta,p.nt); %Adaptation term
    if ismember(lay,[1 2])
       p.o{lay} = zeros(p.ntheta,p.nt);
    end
end

% initial condition: inject random imbalance at the initial time point.
% here, it is done in terms of the imbalance between eyes. can do other random initiation.
rng(1); % set the random seed
p.r{1}(:,1) = rand*.2;
p.r{4}(:,1) = rand*.2;

