clear all; clc; close all;

%% assuming sigma_w = 1, define analytical inverse of noise covariance C_w
N = 20;
M = 3;
r = sym('r','real');
row = repmat(r*eye(N),[1 M]);
C = repmat(row, [M,1]);

idx = 1 : size(C, 1)+1 : numel(C);  %linear indexing for the diagonal
C(idx) = ones(M*N,1);   % put one on the diagonal

C_inv_sym = inv(C);

%% substitute a value for rho, 

r = 0.95;
C_w_inv = double(subs(C_inv_sym));

