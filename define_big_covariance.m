function C = define_big_covariance(M, N, rho, sigma2)
%DEFINE_BIG_COVARIANCE define block partitioned covariance matrix 
% across both time and lines
%
% size of output covariance matrix is [N*M, N*M]
% N is the number of time samples
% M is the number of lines
% rho is correlation among lines
% noises are assumed zero-mean and temporally white
% sigma2 is an optional argument. DEFAULT VARIANCE = 1

if nargin < 4   % sigma2 was not passed
    sigma2 = 1;   % all channels have unit noise variance
end

% C = zeros(N*M);
% for i = 1:M   % fill up off diagonal terms
%     row = repmat(rho*eye(N),[1 M]);
%     row( (i-1)*N+1 : i*N , (i-1)*N+1 : i*N  ) = eye(N);
% end

row = repmat(rho*eye(N),[1 M]);
C = repmat(row, [M,1]);


idx = 1 : size(C, 1)+1 : numel(C);  %linear indexing for the diagonal
C(idx) = ones(M*N,1);   % put one on the diagonal

if sigma2 ~= 1
    C = sigma2*C;
end

end

