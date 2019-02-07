clear; clc;

n_mem = 5;
M = 3;
B = 1:10; % number of training blocks
% OPTINAL add very long training sequence
% B = [B 1000];
alpha = linspace(0,0.99,10); % channels coupling
SNR = -12:3:40;  % var(x) / var(w) in dB
% get corresponding noise variances (assumption is var(x) = 1)
var_w = ( 10.^(SNR/10) ).^(-1);
rho = 0.98;  
num_iter = 1e2;   % number of monte carlo iterations

% start Monte-Carlo simulation
MSE_ls = zeros(numel(SNR),numel(alpha),numel(B));
CRB = zeros(numel(SNR),numel(alpha),numel(B));

for i = 1:numel(var_w)  % index of noise variance
    n_var = var_w(i);
    for j = 1:numel(alpha)  % index of alpha   
        a = alpha(j);
        h = generate_channel(a, n_mem, M);
        for k = 1:numel(B)  % index of num blocks
            b = B(k);
            for iter = 1:num_iter  
                % generate random signals
                X = generate_training_seq(M,b);
                W = generate_noise(M,size(X,2)+n_mem-1,rho,n_var)';                
                
                % unroll into vectors and get block partitioned form
                w = reshape(W',[numel(W) 1]);
                X_conv = generate_block_conv_mat(X',n_mem);
                X_bp = blkdiag(X_conv,X_conv,X_conv);
                y = X_bp*h + w;
                
                % Least-Squares Estimator of H
                h_hat_ls = pinv(X_bp)*y;
                
                err_ls = h_hat_ls - h;
                
                % compute MSE (mean over elements of H_hat and over Monte-Carlo runs)
                MSE_ls(i,j,k) = MSE_ls(i,j,k) + ...
                    mean(err_ls(:).^2)/num_iter;
                
                % compute average CRB using formula that depends on generated data
                crb_mat = inv(X_bp'*X_bp);
                % mean lower bound variance over all parameters
                mean_crb = trace(crb_mat)/size(crb_mat,1);  
                CRB(i,j,k) = CRB(i,j,k) + (mean_crb / num_iter);  % mean over Monte-Carlo runs
                
            end
        end
    end
end
