function X_conv = generate_block_conv_mat(X,n_mem)
%GENERATE_BLOCK_CONV_MAT Summary of this function goes here
%   Detailed explanation goes here

N = size(X,1);
M = size(X,2);

X_conv = zeros(N+n_mem-1, n_mem*M );
for i = 1 : n_mem
    X_conv( i : i+N-1, (i-1)*M+1 : i*M  ) = X;
end


end

