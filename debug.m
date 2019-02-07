% clear; clc; close all;
% 
% n_mem = 5;
% M = 3;
% B = 1:10; % number of training blocks
% % OPTINAL add very long training sequence
% % B = [B 1000];
% alpha = linspace(0,0.99,10); % channels coupling
% SNR = -12:3:40;  % var(x) / var(w) in dB
% % get corresponding noise variances (assumption is var(x) = 1)
% var_w = ( 10.^(SNR/10) ).^(-1);
% rho = 0;   % assume for now uncorrelated noise between lines
% num_iter = 1e2;   % number of monte carlo iterations
% 
% % start Monte-Carlo simulation
% MSE = zeros(numel(SNR),numel(alpha),numel(B));
% 
% for i = 1:numel(var_w)  % index of noise variance
%     n_var = var_w(i);
%     for j = 1:numel(alpha)  % index of alpha   
%         a = alpha(j);
%         h = generate_channel(a, n_mem, M);
%         for k = 1:numel(B)  % index of num blocks
%             b = B(k);
%             for iter = 1:num_iter  
%                 % generate random signals
%                 X = generate_training_seq(M,b);
%                 W = generate_noise(M,size(X,2)+n_mem-1,rho,n_var)';                
%                 
%                 % unroll into vectors and get block partitioned form
%                 w = reshape(W',[numel(W) 1]);
%                 X_conv = generate_block_conv_mat(X',n_mem);
%                 X_bp = blkdiag(X_conv,X_conv,X_conv);
%                 y = X_bp*h; %+ w;
%                 
%                 % Least-Squares Estimator of H
% 
%                 h_hat = pinv(X_bp)*y;
%                 err = h_hat - h;
%                 
%                 % compute MSE (mean over elements of H_hat and over Monte-Carlo runs)
%                 MSE(i,j,k) = MSE(i,j,k) + ...
%                     mean(err(:).^2)/num_iter;
%                 
%             end
%         end
%     end
% end
% 
% %% check convolution
% % istantaneous response of channel
% H_check = toeplitz(a.^(0:M-1)); 
% h_check = H_check(:);
% 
% % time envelope of channels
% env = exp(-(0:n_mem-1)/4);
% 
% H_mem = h_check*env;
% 
% % h11 = H_mem(1,:);
% % h12 = H_mem(2,:);
% % h13 = H_mem(3,:);
% % h21 = H_mem(4,:);
% % h22 = H_mem(5,:);
% % h23 = H_mem(6,:);
% % h31 = H_mem(7,:);
% % h32 = H_mem(8,:);
% % h33 = H_mem(9,:);
% % h33 = H_mem(9,:);
% 
% Y = zeros(3,14);
% for i = 1:3
%     Y(i,:) = conv(X(1,:),H_mem(3*(i-1)+1,:)) + conv(X(2,:),H_mem(3*(i-1)+2,:)) + conv(X(3,:),H_mem(3*(i-1)+3,:));
% end

% clear; clc;
% 
% M = 3;
% alpha = linspace(0,1,10);  % channels coupling
% % remove alpha = 1 to avoid singular matrix H
% alpha(end) = 0.9999;
% 
% SNR = -12:3:40;  % var(x) / var(w) in dB
% % get corresponding noise variances (assumption is var(x) = 1)
% var_w = ( 10.^(SNR/10) ).^(-1);
% rho = 0.1;
% num_iter = 1e2;   % number of monte carlo iterations
% 
% % Define signal to be transmitted
% Ns = 2000;
% X = randn(M,Ns);
% 
% % Define noise covariance across channels
% C_w = rho * ones(M);
% C_w(boolean(eye(M))) = 1;
% 
% % start Monte-Carlo simulation
% MSE_ls = zeros(numel(SNR),numel(alpha));
% MSE_mle = zeros(numel(SNR),numel(alpha));
% MSE_Wiener = zeros(numel(SNR),numel(alpha));
% for i = 1:numel(var_w)  % index of noise variance
%     n_var = var_w(i);
%     C_w = n_var * C_w;   % scale covariance matrix according to noise power
%     
%         for j=1:numel(alpha)
%         % define channel repsonse
%         a = alpha(j);
%         H = toeplitz(a.^(0:M-1)); % true channel response
% 
%             for iter = 1:num_iter  
%                 % generate random gaussian noise (correlated with r)
%                 W = generate_noise(M,Ns,rho,n_var)';                
%                 
%                 % pass signal through MIMO memoryless Channel with AWGN
%                 Y = H*X + W;
%                 
%                 % Least-Squares Estimator of X
%                 X_ls = pinv(H)*Y;
%                 
%                 % MLE Estimator of X
%                 X_ml = (H'*(C_w\H))\(H'*(C_w\Y));
%                 
%                 % Winer Filter (LMMSE)
%                 % X_mmse = inv(H'*inv(C_w)*H + eye(M))*H'*inv(C_w)*Y ;
%                 X_mmse = (H'*(C_w\H)+eye(M))\(H'*(C_w\Y));
%                 
%                 % compute error
%                 err_ls = X_ls - X;
%                 err_mle = X_ml - X;
%                 
%                 % compute MSE (mean over elements of H_hat and over Monte-Carlo runs)
%                 MSE_ls(i,j) = MSE_ls(i,j) + mean(err_ls(:).^2)/num_iter;
%                 MSE_mle(i,j) = MSE_mle(i,j) + mean(err_mle(:).^2)/num_iter;
%                 
%             end
%         end
% end


a = sym('a', 'real');  % alpha
r = sym('r', 'real');  % rho

H = toeplitz([1 a a^2]);
C_w = r*ones(size(H));
C_w(boolean(eye(size(C_w)))) = 1;

CRB_MAT = inv( H' * inv(C_w) * H );
crb = trace(CRB_MAT) / 3;

noise_var = 1./(10.^(SNR/10));

% evaluate at the specific values of alpha and rho
alpha = linspace(0,1,10);
rho = linspace(0,1,10);
alpha(end) = 0.9999;
rho(end) = 0.9999;

% evaluate all possible CRB for combination of values of SNR, alpha and rho
crb_numerical = zeros(numel(noise_var),numel(alpha),numel(rho));
for i = 1 : numel(noise_var)
    var_w = noise_var(i);
    
    for j = 1 : numel(alpha)
        a = alpha(j);
        
        for k = 1 : numel(rho)
            r = rho(k);    
            crb_numerical(i,j,k) = double( var_w * subs(crb) );
            disp(crb_numerical(i,j,k));
        end
    end
end
