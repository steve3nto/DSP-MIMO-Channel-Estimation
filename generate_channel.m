function h_mem = generate_channel( alpha, n_mem, M )
%GENERATE_CHANNEL Summary of this function goes here
%   Detailed explanation goes here

% istantaneous response of channel
H = toeplitz(alpha.^(0:M-1)); 
h = H(:);

% time envelope of channels
env = exp(-(0:n_mem-1)/4);

H_mem = h*env;

h_mem = zeros(n_mem*M^2,1);
for i = 0:M-1
    H_current = H_mem( (M*i+1 : M*(i+1)) , : );
    h_mem( M*n_mem*i+1 : M*n_mem*(i+1)  ) = H_current(:);
end


end

