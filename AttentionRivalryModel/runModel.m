%% Set condition to run

for cond = 1:7
% cond = 1; %Condition to simulate (seven conditions are available here)
%1. Dichoptic gratings / attended   (attended rivalry)
%2. Dichoptic gratings / unattended (unattended rivalry)
%3. Monocular plaids / attended
%4. Monocular plaids / unattended
%5. Swapping with a blank gap before swap (blank-before-swap condition) 
%6. Swapping with no blank (static image condition) 
%7. Swapping with flicker  (flicker-and-swap condition)

%% 
%Setup the stimulus sequence and parameters
p      = setParameters(cond); %set parameters
p      = setStim(p);          %draw stimuli
p      = initTimeSeries(p);   %preallocate data matrices
p.i{1} = p.stimL;             %assign stimulus to the inputs of monocular layers
p.i{2} = p.stimR;             %assign stimulus to the inputs of monocular layers

%Run the model
fprintf('%s / inupt strength: %1.2f %1.2f \n', p.condnames{p.cond}, p.input(1), p.input(2));
p = n_model(p);

keys = {'d', 's', 'r', 'f', 'h', 'o', 'i'};
results = construct_results(p, keys);

save(['matlab_timecourse_cond_', num2str(cond), '.mat'], 'results');

%Plot results
% plotTimeSeries(p);
end


% Addded for my comparrision - Gerion
function [results] = construct_results(p, keys)
%     key = 'r'
%     keys = {'d', 's', 'r', 'f', 'h', 'o', 'i'};
    results = struct();
    for key_id = 1:length(keys)
        key = keys{key_id};
        if strcmp(key, 'o')
            results.(key) = struct();
            results.(key).opponency_left_1 = p.(key){1}(1, :);
            results.(key).opponency_left_2 = p.(key){1}(2, :);
            results.(key).opponency_right_1 = p.(key){2}(1, :);
            results.(key).opponency_right_2 = p.(key){2}(2, :);
        elseif strcmp(key, 'i')
            results.(key).input_left_1 = p.(key){1}(1, :);
            results.(key).input_left_2 = p.(key){1}(2, :);
            results.(key).input_right_1 = p.(key){2}(1, :);
            results.(key).input_right_2 = p.(key){2}(2, :);
        else
            results.(key) = struct();
            results.(key).attention_1 = p.(key){6}(1, :);
            results.(key).attention_2 = p.(key){6}(2, :);
            results.(key).opponency_left_1 = p.(key){4}(1, :);
            results.(key).opponency_left_2 = p.(key){4}(2, :);
            results.(key).opponency_right_1 = p.(key){5}(1, :);
            results.(key).opponency_right_2 = p.(key){5}(2, :);
            results.(key).sensory_left_1 = p.(key){1}(1, :);
            results.(key).sensory_left_2 = p.(key){1}(2, :);
            results.(key).sensory_right_1 = p.(key){2}(1, :);
            results.(key).sensory_right_2 = p.(key){2}(2, :);
            results.(key).summation_1 = p.(key){3}(1, :);
            results.(key).summation_2 = p.(key){3}(2, :);
        end
    end
end
