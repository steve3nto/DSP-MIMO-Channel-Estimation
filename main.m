% DSP HOMEWORK 1
close all; clear; clc; 

%% 1.1) NOISE GENERATION

% DEFINE PARAMETERS
M = 3; % num of lines
N = [10,50,200,10000]; % num of time samples per signal
rho = linspace(0,1,10);

% compute sample covariances for varying rho and M
C_W = cell(length(N), length(rho));
C_true = cell(1, length(rho));

for i = 1:length(N)
    n = N(i);
    for j = 1:length(rho)
        r = rho(j);
        [W, C] = generate_noise(M, n, r);
        % save true covariance matrix
        if isempty(C_true{j})
            C_true{j} = C;
        end
        
        % compute sample covariance
        C_W{i, j} = cov(W);  
    end
end

% figure;
% plotmatrix(W);  % CHECK SCATTER-PLOT

%% CHECK SAMPLE COVARIANCE ESTIMATION ACCURACY 
% use Frobenius norm of C - C_hat to check accuracy
% normalize by ||C|| to get relative error

error = zeros(size(C_W));
for i = 1 : size(C_W,1)   % index of N
    for j = 1 : size(C_W,2)  % index of rho
        C = C_true{1,j};
        C_hat = C_W{i,j};
        error(i,j) = norm(C-C_hat,'fro') / norm(C,'fro');
    end
end

% plot results
figure;
% plot(error', 'o-')

for i = 1 : size(error,1)
    scatter(rho,error(i,:),100,'filled');
    hold on
end
hold off
title('Effect of sample size in covariance estimation');
% legend(string(N),'Location','best');
hleg = legend(string(N),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','N Time Samples')
xlabel('Correlation Coefficient \rho');
ylabel('Relative Frobenius Norm of Error Matrix');
xticks(rho);
grid on;


%% (EXTRA) CHECK NOISE IS TEMPORALLY WHITE
M_check = 3;
N_check = 10000;
rho_check = 0.6;
W = generate_noise(M_check, N_check, rho_check);
% compute autocorrelation for each channel noise
R = zeros(2*N_check-1,M_check);
for i=1:M_check
    [rho,lags] = xcorr(W(:,i));
    R(:,i) = rho;
    % OPTIONAL, normalize with r(0), which is max value
    % R(:,i) = R(:,i) ./ r(lags==0);
end
 
% plot the autocorrelations around 0 lag
N_lags = 10;
center = find(lags==0);
lag_idx = center - N_lags : center + N_lags;
figure;
for i=1:M_check
    stem(-N_lags:N_lags, R(lag_idx,i), 'LineWidth',1.2, 'MarkerSize', 7);
    hold on
end
hold off
xlabel('lag')
ylabel('Correlation value')
title('Autocorrelation plot to check temporal whiteness')
legend TOGGLE

%% 1.2) MIMO FILTER ESTIMATION (MEMORYLESS)
clear; clc; 

M = 3;
B = 1:10; % number of training blocks
% OPTINAL add very long training sequence
% B = [B 1000];
alpha = linspace(0,0.99,10); % channels coupling
SNR = -12:3:40;  % var(x) / var(w) in dB
% get corresponding noise variances (assumption is var(x) = 1)
var_w = ( 10.^(SNR/10) ).^(-1);
rho = 0.999;   
num_iter = 8e2;   % number of monte carlo iterations

% start Monte-Carlo simulation
MSE_ls = zeros(numel(SNR),numel(alpha),numel(B));
MSE_mle = zeros(numel(SNR),numel(alpha),numel(B));
CRB = zeros(numel(SNR),numel(alpha),numel(B));

for i = 1:numel(var_w)  % index of noise variance
    n_var = var_w(i);
    for j = 1:numel(alpha)  % index of alpha   
        a = alpha(j);
        H = toeplitz(a.^(0:M-1)); % true channel response
        for k = 1:numel(B)  % index of num blocks
            b = B(k);
            for iter = 1:num_iter  
                % generate random signals
                X = generate_training_seq(M,b);
                N = size(X,2);
                W = generate_noise(M,N,rho,n_var)';                
                
                % unroll into vectors and get block partitioned form
                h = H(:);
                w = reshape(W',[numel(W) 1]);
                X_bp = blkdiag(X',X',X');
                y = X_bp*h + w;
                
                % Least-Squares Estimator of H
                h_hat_ls = pinv(X_bp)*y;
                % MLE Estimator of H
                C_w = define_big_covariance(M,N,rho);
                C_w = n_var * C_w;   % scale covariance matrix according to noise power
                h_hat_mle = (X_bp'*(C_w\X_bp))\(X_bp'*(C_w\y));
                
                % compute error
                err_ls = h_hat_ls - h;
                err_mle = h_hat_mle - h;
                
                % compute MSE (mean over elements of H_hat and over Monte-Carlo runs)
                MSE_ls(i,j,k) = MSE_ls(i,j,k) + mean(err_ls(:).^2)/num_iter;
                MSE_mle(i,j,k) = MSE_mle(i,j,k) + mean(err_mle(:).^2)/num_iter;
                
                % compute average CRB using formula that depends on generated data
                crb_mat = n_var .* inv(X_bp'*X_bp);
                % mean lower bound variance over all parameters
                mean_crb = trace(crb_mat)/size(crb_mat,1);  
                CRB(i,j,k) = CRB(i,j,k) + (mean_crb / num_iter);  % mean over Monte-Carlo runs
                
            end
        end
    end
end

%% compute theoretical (analytical) CRB as a function of B and SNR

% use a finer scale for theoretical CRB
SNR_CRB = -12:1:40;  % var(x) / var(w) in dB
% get corresponding noise variances (assumption is var(x) = 1)
var_w_CRB = ( 10.^(SNR_CRB/10) ).^(-1);

% analytcial MSE_CRB is var(w) / 10*B for all combinations of SNR and B 
MSE_CRB = (repmat(var_w_CRB,[numel(B),1]) ./ (20.*repmat(B',[1,numel(var_w_CRB)]) ) )';

%% plot MSE vs SNR in dB scale
mse_ls_alpha = squeeze(mean(MSE_ls,3)); %average away effect of B
mse_ls_B = squeeze(mean(MSE_ls,2));    % average away effect of alpha
mse_mle_alpha = squeeze(mean(MSE_mle,3)); 
mse_mle_B = squeeze(mean(MSE_mle,2));    


% plot results to see effect of alpha (coupling coefficient)
figure;
semilogy(repmat(SNR',[1 size(mse_ls_alpha,2)]),mse_ls_alpha, 'o-')
title('Effect of Channel Coupling on Channel Estimation (LS Estimator)');
hleg = legend(string(alpha),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','Coupling Coefficient')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

% plot results to see effect of B (Number of block repetitions)
figure;
semilogy(repmat(SNR',[1 size(mse_ls_B,2)]),mse_ls_B, 'o-')
title('Effect of Training Sequence Length on Channel Estimation (LS Estimator)');
hleg = legend(string(B),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','B - Training Block Repetitions')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

% plot results to see effect of alpha (coupling coefficient)
figure;
semilogy(repmat(SNR',[1 size(mse_ls_alpha,2)]),mse_mle_alpha, 'o-')
title('Effect of Channel Coupling on Channel Estimation (ML Estimator)');
hleg = legend(string(alpha),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','Coupling Coefficient')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

% plot results to see effect of B (Number of block repetitions)
figure;
semilogy(repmat(SNR',[1 size(mse_ls_B,2)]),mse_mle_B, 'o-')
title('Effect of Training Sequence Length on Channel Estimation (ML Estimator)');
hleg = legend(string(B),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','B - Training Block Repetitions')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

% plot results to see effect of B (analytical CRB case)
figure;
semilogy(repmat(SNR_CRB',[1 size(MSE_CRB,2)]),MSE_CRB, 'o-')
title('Effect of Training Sequence Length on Channel Estimation (CRB)');
hleg = legend(string(B),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','B - Training Block Repetitions')
xlabel('SNR [dB]');
ylabel('Analytical Average MSE');
xticks(SNR_CRB);
grid on; 

%% COMPARE CRB to ESTIMATED IN SINGLE GRAPH
b_min = B(1);    % plot for shorter training sequence
b_mid = B(floor(length(B)/2));
b_max = B(end);  % plot for longer training sequence

a_min = alpha(1);
a_mid = alpha(floor(length(B)/2));
a_max = alpha(end);

MSE_ls_bmin = mse_ls_B(:,1);
MSE_ls_bmid = mse_ls_B(:,floor(length(B)/2));
MSE_ls_bmax = mse_ls_B(:,end);
MSE_ls_amin = mse_ls_alpha(:,1);
MSE_ls_amid = mse_ls_alpha(:,floor(length(B)/2));
MSE_ls_amax = mse_ls_alpha(:,end);

% analytcial MSE_CRB is var(w) / 20*B for all combinations of SNR and B 
MSE_CRB_bmin = (var_w ./ (20*b_min))';
MSE_CRB_bmid = (var_w ./ (20*b_mid))';
MSE_CRB_bmax = (var_w ./ (20*b_max))';

% MSE_CRB_averaged over B values
MSE_CRB_alpha = mean(MSE_CRB,2);

% define lines width and markers size
lw = 1.2;
ms = 9;

figure;
semilogy(SNR, MSE_CRB_bmin, '--','LineWidth',lw);
hold on;
semilogy(SNR, MSE_CRB_bmid, '--','LineWidth',lw);
semilogy(SNR, MSE_CRB_bmax, '--','LineWidth',lw);
semilogy(SNR, MSE_ls_bmin, '*','MarkerSize', ms);
semilogy(SNR, MSE_ls_bmid, '*','MarkerSize', ms);
semilogy(SNR, MSE_ls_bmax, '*','MarkerSize', ms);
title('MSE vs CRB. Guassian Training with \rho = 0.999');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
hleg = legend(string(b_min)+' CRB',string(b_mid)+' CRB',string(b_max)+' CRB',...
                        string(b_min), string(b_mid), string(b_max));
htitle = get(hleg,'Title');
set(htitle,'String','B Block Repetitions')
grid on; 
hold off;

figure;
semilogy(SNR_CRB, MSE_CRB_alpha, '--','LineWidth', lw);
hold on;
semilogy(SNR, MSE_ls_amin, '*','MarkerSize', ms);
semilogy(SNR, MSE_ls_amid, '*','MarkerSize', ms);
semilogy(SNR, MSE_ls_amax, '*','MarkerSize', ms);
title('MSE vs CRB. Guassian Training with \rho = 0.999');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
hleg = legend('CRB',string(a_min),string(a_mid),string(a_max));
htitle = get(hleg,'Title');
set(htitle,'String','\alpha Channel Coupling')
grid on; 
hold off;



%% repeat for binary valued training signals
clc; 

M = 3;
B = 1:10; % number of training blocks
% OPTINAL add very long training sequence
% B = [B 1000];
alpha = 0.5;  % channels coupling
SNR = -12:3:40;  % var(x) / var(w) in dB
% get corresponding noise variances (assumption is var(x) = 1)
var_w = ( 10.^(SNR/10) ).^(-1);
rho = 0.999;   
num_iter = 8e2;   % number of monte carlo iterations

% start Monte-Carlo simulation
MSE_ls_binary = zeros(numel(SNR),numel(B));
MSE_mle_binary = zeros(numel(SNR),numel(B));

for i = 1:numel(var_w)  % index of noise variance
        n_var = var_w(i);
        H = toeplitz(alpha.^(0:M-1)); % true channel response
        for k = 1:numel(B)  % index of num blocks
            b = B(k);
            for iter = 1:num_iter  
                % generate random signals
                X = generate_training_seq_binary(M,b);
                N = size(X,2);
                W = generate_noise(M,N,rho,n_var)';                
                
                % unroll into vectors and get block partitioned form
                h = H(:);
                w = reshape(W',[numel(W) 1]);
                X_bp = blkdiag(X',X',X');
                y = X_bp*h + w;
                
                % Least-Squares Estimator of H
                h_hat_ls = pinv(X_bp)*y;
                % MLE Estimator of H
                C_w = define_big_covariance(M,N,rho);
                C_w = n_var * C_w;   % scale covariance matrix according to noise power
                h_hat_mle = (X_bp'*(C_w\X_bp))\(X_bp'*(C_w\y));
                
                % compute error
                err_ls = h_hat_ls - h;
                err_mle = h_hat_mle - h;
                
                % compute MSE (mean over elements of H_hat and over Monte-Carlo runs)
                MSE_ls_binary(i,k) = MSE_ls_binary(i,k) + mean(err_ls(:).^2)/num_iter;
                MSE_mle_binary(i,k) = MSE_mle_binary(i,k) + mean(err_mle(:).^2)/num_iter;
                
            end
        end
end

%% Plot results comparing with CRB
b_min = B(1);    % plot for shorter training sequence
b_mid = B(floor(length(B)/2));
b_max = B(end);  % plot for longer training sequence

MSE_ls_bmin_bin = MSE_ls_binary(:,1);
MSE_ls_bmid_bin = MSE_ls_binary(:,floor(length(B)/2));
MSE_ls_bmax_bin = MSE_ls_binary(:,end);

% analytcial MSE_CRB is var(w) / 20*B for all combinations of SNR and B 
MSE_CRB_bmin = (var_w ./ (20*b_min))';
MSE_CRB_bmid = (var_w ./ (20*b_mid))';
MSE_CRB_bmax = (var_w ./ (20*b_max))';

figure;
semilogy(SNR, MSE_CRB_bmin, '--','LineWidth', lw);
hold on;
semilogy(SNR, MSE_CRB_bmid, '--','LineWidth', lw);
semilogy(SNR, MSE_CRB_bmax, '--','LineWidth', lw);
semilogy(SNR, MSE_ls_bmin_bin, '*', 'MarkerSize', ms);
semilogy(SNR, MSE_ls_bmid_bin, '*', 'MarkerSize', ms);
semilogy(SNR, MSE_ls_bmax_bin, '*', 'MarkerSize', ms);
title('MSE vs CRB. Binary Training with \rho = 0.999');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
hleg = legend(string(b_min)+' CRB',string(b_mid)+' CRB',string(b_max)+' CRB',...
                        string(b_min), string(b_mid), string(b_max));
htitle = get(hleg,'Title');
set(htitle,'String','B Block Repetitions')
grid on; 
hold off;

%% plot gaussian vs binary in single graph

figure;
semilogy(SNR, MSE_ls_bmin,'LineWidth', lw);
hold on;
semilogy(SNR, MSE_ls_bmid,'LineWidth', lw);
semilogy(SNR, MSE_ls_bmax,'LineWidth', lw);
semilogy(SNR, MSE_ls_bmin_bin, '*', 'MarkerSize', ms, 'LineWidth', lw);
semilogy(SNR, MSE_ls_bmid_bin, '*', 'MarkerSize', ms, 'LineWidth', lw);
semilogy(SNR, MSE_ls_bmax_bin, '*', 'MarkerSize', ms, 'LineWidth', lw);
title('Comparison Binary vs Gaussian. \rho = 0.999, MC Runs = 10');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
hleg = legend(string(b_min)+' Gaussian',string(b_mid)+' Gaussian',string(b_max)+' Gaussian',...
                        string(b_min)+ ' Binary', string(b_mid)+' Binary', string(b_max)+' Binary');
htitle = get(hleg,'Title');
set(htitle,'String','B Block Repetitions')
grid on; 
hold off;

%% 1.3) MIMO FILTER ESTIMATION (MEMORY FILTERS)
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
rho = 0.999;  
num_iter = 1e3;   % number of monte carlo iterations

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
                crb_mat = n_var .* inv(X_bp'*X_bp);
                % mean lower bound variance over all parameters
                mean_crb = trace(crb_mat)/size(crb_mat,1);  
                CRB(i,j,k) = CRB(i,j,k) + (mean_crb / num_iter);  % mean over Monte-Carlo runs
                
            end
        end
    end
end

%% visualize CRB matrix
figure;
imagesc(crb_mat);
title('CRB matrix of memory channel parameters');
xlabel('memory channel parameters');
ylabel('memory channel parameters');

%% plot MSE vs SNR at varying B and alpha
mse_ls_alpha = squeeze(mean(MSE_ls,3)); %average away effect of B
mse_ls_B = squeeze(mean(MSE_ls,2));    % average away effect of alpha

% TO-DO change MSE to log scale (or MSE-dB)

% plot results to see effect of alpha (coupling coefficient)
figure;
semilogy(repmat(SNR',[1 size(mse_ls_alpha,2)]),mse_ls_alpha, 'o-')
title('Effect of Channel Coupling on Channel Estimation');
hleg = legend(string(alpha),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','Coupling Coefficient')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

% plot results to see effect of B (Number of block repetitions)
figure;
semilogy(repmat(SNR',[1 size(mse_ls_B,2)]),mse_ls_B, 'o-')
title('Effect of Training Sequence Length on Channel Estimation');
hleg = legend(string(B),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','B - Training Block Repetitions')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

%% COMPARE CRB to ESTIMATED IN SINGLE GRAPH
b_min = B(1);    % plot for shorter training sequence
b_mid = B(floor(length(B)/2));
b_max = B(end);  % plot for longer training sequence

a_min = alpha(1);
a_mid = alpha(floor(length(B)/2));
a_max = alpha(end);

MSE_ls_bmin = mse_ls_B(:,1);
MSE_ls_bmid = mse_ls_B(:,floor(length(B)/2));
MSE_ls_bmax = mse_ls_B(:,end);
MSE_ls_amin = mse_ls_alpha(:,1);
MSE_ls_amid = mse_ls_alpha(:,floor(length(alpha)/2));
MSE_ls_amax = mse_ls_alpha(:,end);

% analytcial MSE_CRB is var(w) / 20*B for all combinations of SNR and B 
MSE_CRB_alpha = squeeze(mean(CRB,3)); %average away effect of B
MSE_CRB_B = squeeze(mean(CRB,2));    % average away effect of alpha

MSE_CRB_bmin = MSE_CRB_B(:,1);
MSE_CRB_bmid = MSE_CRB_B(:,floor(length(B)/2));
MSE_CRB_bmax = MSE_CRB_B(:,end);

% CRB does not depend on alpha, average together
MSE_CRB_a = mean(MSE_CRB_alpha,2);

% define line width and marker size
lw = 1.2;
ms = 9;

% write cell array of chars for the legend
legend_cell = {char(string(a_min)), char(string(a_mid)), char(string(a_max))};

figure;
semilogy(SNR, MSE_CRB_bmin, '--k','LineWidth',lw);
hold on;
semilogy(SNR, MSE_CRB_bmid, '--k','LineWidth',lw);
semilogy(SNR, MSE_CRB_bmax, '--k','LineWidth',lw);
h1 = semilogy(SNR, MSE_ls_bmin, '*','MarkerSize',ms);
h2 = semilogy(SNR, MSE_ls_bmid, '*','MarkerSize',ms);
h3 = semilogy(SNR, MSE_ls_bmax, '*','MarkerSize',ms);
title('Memory Channel Estimation - Gaussian Training Sequence');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on; 
hleg = legend([h1,h2,h3],legend_cell,'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','B')
hold off;

figure;
semilogy(SNR, MSE_CRB_a, '--k','LineWidth',lw);
hold on;
semilogy(SNR, MSE_ls_amin, '*','MarkerSize',ms);
semilogy(SNR, MSE_ls_amid, '*','MarkerSize',ms);
semilogy(SNR, MSE_ls_amax, '*','MarkerSize',ms);
title('Memory Channel Estimation - Gaussian Training Sequence');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
hleg = legend('CRB',string(a_min),string(a_mid),string(a_max));
htitle = get(hleg,'Title');
set(htitle,'String','\alpha - Channel Coupling')
grid on; 
hold off;


%% 1.4) MIMO DECONVOLUTION

clear; clc;

% plot CRB  (calculated with symbolic toolbox)

a = sym('a', 'real');  % alpha
r = sym('r', 'real');  % rho

H = toeplitz([1 a a^2]);
C_w = r*ones(size(H));
C_w(boolean(eye(size(C_w)))) = 1;

CRB_MAT = inv( H' * inv(C_w) * H );
crb = trace(CRB_MAT) / 3;

SNR = -12:3:40;  % var(x) / var(w) in dB
noise_var = 1./(10.^(SNR/10));

% evaluate at the specific values of alpha and rho
alpha = linspace(0,1,10);
rho_vec = linspace(0,1,10);
alpha(end) = 0.9999;
rho_vec(end) = 0.9999;

% evaluate all possible CRB for combination of values of SNR, alpha and rho
crb_numerical = zeros(numel(noise_var),numel(alpha),numel(rho_vec));
for i = 1 : numel(noise_var)
    var_w = noise_var(i);
    
    for j = 1 : numel(alpha)
        a = alpha(j);
        
        for k = 1 : numel(rho_vec)
            r = rho_vec(k);    
            crb_numerical(i,j,k) = double( var_w * subs(crb) );
        end
    end
end

% average away effect of rho
crb_num_alpha = squeeze(mean(crb_numerical,3)); %average away effect of rho
crb_num_rho = squeeze(mean(crb_numerical,2));    % average away effect of alpha

figure;
semilogy(repmat(SNR',[1 size(crb_num_rho,2)]),crb_num_rho, 'o-')
title('Effect of \rho (Noise Correlation) on Deconvolution (CRB THEORETICAL OPTIMUM)');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
legend(string(rho_vec));
grid on;

figure;
semilogy(repmat(SNR',[1 size(crb_num_alpha,2)]),crb_num_alpha, 'o-')
title('Effect of \alpha (Channel Coupling) on Deconvolution (CRB THEORETICAL OPTIMUM)');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
legend(string(alpha));
grid on;

%% Evaluate analytical CRB at wanted values of rho and alpha
% evaluate all possible CRB for combination of values of SNR, alpha and rho

% fix alpha at 0.4
a = 0.4;
crb_num_fix_alpha = zeros(numel(noise_var),numel(rho_vec));
for i = 1 : numel(noise_var)
    var_w = noise_var(i);
    
    for j = 1 : numel(rho_vec)
                
        r = rho_vec(j);    
        crb_num_fix_alpha(i,j) = double( var_w * subs(crb) );
        
    end
end

crb_num_fix_rho = zeros(numel(noise_var),numel(alpha));
% fix rho to 0.1
r = 0.1;
for i = 1 : numel(noise_var)
    var_w = noise_var(i);
    
    for j = 1 : numel(alpha)
                
        a = alpha(j);    
        crb_num_fix_rho(i,j) = double( var_w * subs(crb) );
        
    end
end

%% we fix alpha and analyze for different values of rho

M = 3;
alpha = 0.4;  % channels coupling
SNR = -12:3:40;  % var(x) / var(w) in dB
% get corresponding noise variances (assumption is var(x) = 1)
var_w = ( 10.^(SNR/10) ).^(-1);
rho_vec = linspace(0,1,10);
% remove rho = 1 to avoid singular matrix C_w
rho_vec(end) = 0.9999;

num_iter = 1e3;   % number of monte carlo iterations

% memoryless channel reponse
H = toeplitz(alpha.^(0:M-1)); % true channel response
% Define signal to be transmitted
Ns = 2000;
X = randn(M,Ns);

% start Monte-Carlo simulation
MSE_ls = zeros(numel(SNR),numel(rho_vec));
MSE_mle = zeros(numel(SNR),numel(rho_vec));
MSE_Wiener = zeros(numel(SNR),numel(rho_vec));
for i = 1:numel(var_w)  % index of noise variance
    n_var = var_w(i);
        for j=1:numel(rho_vec)
        r = rho_vec(j);
        % define noise covariance across lines
        C_w = r * ones(M);
        C_w(boolean(eye(M))) = 1;
        C_w = n_var * C_w;   % scale covariance matrix according to noise power

            for iter = 1:num_iter  
                % generate random gaussian noise (correlated with r)
                W = generate_noise(M,Ns,r,n_var)';                
                
                % pass signal through MIMO memoryless Channel with AWGN
                Y = H*X + W;
                
                % Least-Squares Estimator of X
                X_ls = pinv(H)*Y;
                
                % MLE Estimator of X
                X_ml = (H'*(C_w\H))\(H'*(C_w\Y));
                
                % Winer Filter (LMMSE)
                % X_mmse = inv(H'*inv(C_w)*H + eye(M))*H'*inv(C_w)*Y ;
                X_mmse = (H'*(C_w\H)+eye(M))\(H'*(C_w\Y));
                
                % compute error
                err_ls = X_ls - X;
                err_mle = X_ml - X;
                err_mmse = X_mmse - X;
                
                % compute MSE (mean over elements of H_hat and over Monte-Carlo runs)
                MSE_ls(i,j) = MSE_ls(i,j) + mean(err_ls(:).^2)/num_iter;
                MSE_mle(i,j) = MSE_mle(i,j) + mean(err_mle(:).^2)/num_iter;
                MSE_Wiener(i,j) = MSE_Wiener(i,j) + mean(err_mmse(:).^2)/num_iter;
                
            end
        end
end

%% Plot results

% plot LS results to see effect of rho (AWGN correlation coefficient)
figure;
semilogy(repmat(SNR',[1 size(MSE_ls,2)]),MSE_ls, 'o-')
title('Effect of \rho (Noise Correlation) on Deconvolution (LS Estimator)');
hleg = legend(string(rho_vec),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','\rho - AWGN correlation')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

% plot MLE results to see effect of rho (AWGN correlation coefficient)
figure;
semilogy(repmat(SNR',[1 size(MSE_mle,2)]),MSE_mle, 'o-')
title('Effect of \rho (Noise Correlation) on Deconvolution (ML Estimator)');
hleg = legend(string(rho_vec),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','\rho - AWGN correlation')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

% plot MMSE results to see effect of rho (AWGN correlation coefficient)
figure;
semilogy(repmat(SNR',[1 size(MSE_Wiener,2)]),MSE_Wiener, 'o-')
title('Effect of \rho (Noise Correlation) on Deconvolution (LMMSE Estimator)');
hleg = legend(string(rho_vec),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','\rho - AWGN correlation')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;



%% we fix rho and analyze for different values of alpha
clc;

M = 3;
alpha = linspace(0,1,10);  % channels coupling
% remove alpha = 1 to avoid singular matrix H
alpha(end) = 0.9999;

SNR = -12:3:40;  % var(x) / var(w) in dB
% get corresponding noise variances (assumption is var(x) = 1)
var_w = ( 10.^(SNR/10) ).^(-1);
rho = 0.1;
num_iter = 1e2;   % number of monte carlo iterations

% Define signal to be transmitted
Ns = 2000;
X = randn(M,Ns);

% Define noise covariance across channels
C_w = rho * ones(M);
C_w(boolean(eye(M))) = 1;

% start Monte-Carlo simulation
MSE_ls_a = zeros(numel(SNR),numel(alpha));
MSE_mle_a = zeros(numel(SNR),numel(alpha));
MSE_Wiener_a = zeros(numel(SNR),numel(alpha));
for i = 1:numel(var_w)  % index of noise variance
    n_var = var_w(i);
    C_w = n_var * C_w;   % scale covariance matrix according to noise power
    
        for j=1:numel(alpha)
        % define channel repsonse
        a = alpha(j);
        H = toeplitz(a.^(0:M-1)); % true channel response

            for iter = 1:num_iter  
                % generate random gaussian noise (correlated with r)
                W = generate_noise(M,Ns,rho,n_var)';                
                
                % pass signal through MIMO memoryless Channel with AWGN
                Y = H*X + W;
                
                % Least-Squares Estimator of X
                X_ls = pinv(H)*Y;
                
                % MLE Estimator of X
                X_ml = (H'*(C_w\H))\(H'*(C_w\Y));
                
                % Winer Filter (LMMSE)
                % X_mmse = inv(H'*inv(C_w)*H + eye(M))*H'*inv(C_w)*Y ;
                X_mmse = (H'*(C_w\H)+eye(M))\(H'*(C_w\Y));
                
                
                % compute error
                err_ls = X_ls - X;
                err_mle = X_ml - X;
                err_mmse = X_mmse - X;
                
                % compute MSE (mean over elements of H_hat and over Monte-Carlo runs)
                MSE_ls_a(i,j) = MSE_ls_a(i,j) + mean(err_ls(:).^2)/num_iter;
                MSE_mle_a(i,j) = MSE_mle_a(i,j) + mean(err_mle(:).^2)/num_iter;
                MSE_Wiener_a(i,j) = MSE_Wiener_a(i,j) + mean(err_mmse(:).^2)/num_iter;
                
            end
        end
end

%% Plot results

% plot LS results to see effect of rho (AWGN correlation coefficient)
figure;
semilogy(repmat(SNR',[1 size(MSE_ls_a,2)]),MSE_ls_a, 'o-')
title('Effect of \alpha (channels coupling) on Deconvolution (LS Estimator)');
hleg = legend(string(alpha),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','\alpha')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

% plot MLE results to see effect of rho (AWGN correlation coefficient)
figure;
semilogy(repmat(SNR',[1 size(MSE_mle_a,2)]),MSE_mle_a, 'o-')
title('Effect of \alpha (Channel Coupling) on Deconvolution (ML Estimator)');
hleg = legend(string(alpha),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','\alpha')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

% plot MMSE results to see effect of rho (AWGN correlation coefficient)
figure;
semilogy(repmat(SNR',[1 size(MSE_Wiener_a,2)]),MSE_Wiener_a, 'o-')
title('Effect of \alpha (Channel Coupling) on Deconvolution (LMMSE Estimator)');
hleg = legend(string(alpha),'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','\alpha')
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on;

%% Compare on a single graph the effect of rho
% with alpha = 0.4

r_min = rho_vec(1);    % plot for shorter training sequence
r_mid = rho_vec(floor(length(rho_vec)/2));
r_max = rho_vec(end);  % plot for longer training sequence

% LS Estimator
MSE_ls_rmin = MSE_ls(:,1);
MSE_ls_rmid = MSE_ls(:,floor(length(rho_vec)/2));
MSE_ls_rmax = MSE_ls(:,end);

% MMSE Estimator
MSE_mmse_rmin = MSE_Wiener(:,1);
MSE_mmse_rmid = MSE_Wiener(:,floor(length(rho_vec)/2));
MSE_mmse_rmax = MSE_Wiener(:,end);

% CRB as function of SNR and rho, with fixed alpha = 0.4;
MSE_CRB_rmin = crb_num_fix_alpha(:,1);
MSE_CRB_rmid = crb_num_fix_alpha(:,floor(length(rho_vec)/2));
MSE_CRB_rmax = crb_num_fix_alpha(:,end);

% define line width and marker size
lw = 1.2;
ms = 9;

% write cell array of chars for the legend
legend_cell = {char('LS '+string(r_min)),char('MMSE '+string(r_min)),...
    char('LS '+string(r_mid)),char('MMSE '+string(r_mid)),...
    char('LS '+string(r_max)), char('MMSE'+string(r_max))};

figure;
semilogy(SNR, MSE_CRB_rmin, '--k','LineWidth',lw);
hold on;
semilogy(SNR, MSE_CRB_rmid, '--k','LineWidth',lw);
semilogy(SNR, MSE_CRB_rmax, '--k','LineWidth',lw);
h1 = semilogy(SNR, MSE_ls_rmin, '*','MarkerSize',ms);
h2 = semilogy(SNR, MSE_ls_rmid, '*','MarkerSize',ms);
h3 = semilogy(SNR, MSE_ls_rmax, '*','MarkerSize',ms);
h4 = semilogy(SNR, MSE_mmse_rmin, '-o','MarkerSize',ms);
h5 = semilogy(SNR, MSE_mmse_rmid, '-o','MarkerSize',ms);
h6 = semilogy(SNR, MSE_mmse_rmax, '-o','MarkerSize',ms);
title('MIMO Deconvolution Performance');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on; 
hleg = legend([h1,h4,h2,h5,h3,h6],legend_cell,'Location','NE');
htitle = get(hleg,'Title');
set(htitle,'String','\rho')
hold off;


%% Compare on a single graph the effect of alpha
% with rho = 0.1

a_min = alpha(1);    % plot for shorter training sequence
a_mid = alpha(floor(length(rho_vec)/2));
a_max = alpha(end);  % plot for longer training sequence

% LS Estimator
MSE_ls_amin = MSE_ls_a(:,1);
MSE_ls_amid = MSE_ls_a(:,floor(length(rho_vec)/2));
MSE_ls_amax = MSE_ls_a(:,end);

% MMSE Estimator
MSE_mmse_amin = MSE_Wiener_a(:,1);
MSE_mmse_amid = MSE_Wiener_a(:,floor(length(rho_vec)/2));
MSE_mmse_amax = MSE_Wiener_a(:,end);

% CRB as function of SNR and rho, with fixed alpha = 0.4;
MSE_CRB_amin = crb_num_fix_rho(:,1);
MSE_CRB_amid = crb_num_fix_rho(:,floor(length(rho_vec)/2));
MSE_CRB_amax = crb_num_fix_rho(:,end);

% define line width and marker size
lw = 1.2;
ms = 9;

% write cell array of chars for the legend
legend_cell = {char('LS '+string(a_min)),char('MMSE '+string(a_min)),...
    char('LS '+string(a_mid)),char('MMSE '+string(a_mid)),...
    char('LS '+string(a_max)), char('MMSE'+string(a_max))};

figure;
semilogy(SNR, MSE_CRB_amin, '--k','LineWidth',lw);
hold on;
semilogy(SNR, MSE_CRB_amid, '--k','LineWidth',lw);
semilogy(SNR, MSE_CRB_amax, '--k','LineWidth',lw);
h1 = semilogy(SNR, MSE_ls_amin, '*','MarkerSize',ms);
h2 = semilogy(SNR, MSE_ls_amid, '*','MarkerSize',ms);
h3 = semilogy(SNR, MSE_ls_amax, '*','MarkerSize',ms);
h4 = semilogy(SNR, MSE_mmse_amin, '-o','MarkerSize',ms);
h5 = semilogy(SNR, MSE_mmse_amid, '-o','MarkerSize',ms);
h6 = semilogy(SNR, MSE_mmse_amax, '-o','MarkerSize',ms);
title('MIMO Deconvolution Performance');
xlabel('SNR [dB]');
ylabel('Average MSE');
xticks(SNR);
grid on; 
hleg = legend([h1,h4,h2,h5,h3,h6],legend_cell,'Location','best');
htitle = get(hleg,'Title');
set(htitle,'String','\alpha')
hold off;