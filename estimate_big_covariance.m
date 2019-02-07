function C = estimate_big_covariance(M, N, rho, sigma2, mu)
%ESTIMATE_BIG_COVARIANCE estimate covariance matrix across both time and lines
%   size of output covariance matrix is [N*M, N*M]
% N is the number of time samples
% M is the number of lines
% rho is correlation among lines
% noises are assumed temporally white
% mu and sigma2 are optional arguments. DEFAULT: 0 mean and 1 variance

if nargin < 4   % mu and sigma were not passed
    mu = zeros(1,M);    % zero mean for all channels
    sigma2 = 1;   % all channels have unit noise variance

elseif nargin < 5   % mu was not passed
    mu = zeros(1,M);    % zero mean for all channels
end

N_realizations = 1e5;
W = zeros(N_realizations, M*N);
for i = 1 : N_realizations
    w = generate_noise(M, N, rho, sigma2, mu);   % this is MxN
    W(i,:) = w(:)';
end

C = cov(W);

end

