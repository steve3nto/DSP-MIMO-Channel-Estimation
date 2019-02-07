function X = generate_training_seq_binary(M, B)
%GENERATE_TRAINING_SEQ_BINARY Summary of this function goes here
%   Detailed explanation goes here

%GENERATE_TRAINING_SEQ Binary sequence with 0 mean and unit variance
% Generates training sequence for M channels and 
% B blocks of length 20
% Output X has size M x (20*B)
% for simulation purposes the variance of training sequence is set
% to 1, and we will vary the noise variance to get wanted SNR

% different sequence for each channel
X = zeros(M,20*B);
for i= 1 : M  % channel index
    x = 2*randi([0 1],20,1) - 1; % single period of 20 samples
    x_per = x * ones(1,B);
    x = x_per(:)';
    X(i,:) = x;
end



end

