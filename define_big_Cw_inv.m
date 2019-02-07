function C_inv = define_big_Cw_inv(M, N, rho, sigma2)
% Defines inverse of noise covariance across both time and channels
% using the analytical formula
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

row = repmat(-rho*eye(N),[1 M]);
C_inv = repmat(row, [M,1]);


idx = 1 : size(C_inv, 1)+1 : numel(C_inv);  %linear indexing for the diagonal
C_inv(idx) = 1+rho;   % put one on the diagonal

C_inv = C_inv / (1 + rho - 2*rho^2 );

if sigma2 ~= 1
    C_inv = C_inv / sigma2;
end

end


