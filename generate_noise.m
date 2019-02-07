function [W, C] = generate_noise(M, N, rho, sigma2, mu)
%GENERATE_NOISE generate gaussian correlated noise among M lines
% INPUTS:
% M: number of lines (signals)
% N: number of time samples per line
% rho: correlation coefficient between adiacient lines
% noise correlaton is rho^abs(i-j), where i and j are line indices
% sigma2: optional variance power, default = 1
% mu: optinal 1xM vector of mean values. Default is 0 mean
% OUTPUTS:
% W: NxM matrix of temporally white gaussian noise across M lines
% C: MxM matrix, true covariance of the noise between the M lines
%
if nargin < 4   % mu and sigma were not passed
    mu = zeros(1,M);    % zero mean for all channels
    sigma2 = 1;   % all channels have unit noise variance

elseif nargin < 5   % mu was not passed
    mu = zeros(1,M);    % zero mean for all channels
end

% noise shall be correlated
% c = sigma2 * rho .^ (0:M-1);
c = sigma2 * [1, rho*ones(1,M-1)];
% set to zero far away variables
%c(3:M) = 0; 
C = toeplitz(c);   % MxM covariance matrix across channels
 
W = mvnrnd(mu,C,N);  % sample N times from M-D gaussian

end

