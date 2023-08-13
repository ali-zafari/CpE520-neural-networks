close all
clc
clear all
%{
X1 = [2 1 3; 1 2 1]
%rref(X1)
X2 = [2 1; 1 2; 3 1];
X3 = [1 3 2 5; 2 5 3 9; 2 1 -1 5; 3 2 -1 8; 1 1 0 3];
X4 = [3 2 2; 2 3 -2];

%rank(X1'*X1)

[V, D]=eig(X1'*X1)

A = double(X1'*X1 - (-65^(1/2) + 10)*eye(3))

[L U] = lu(A)
rank(A)

N = null(A)

[U, S, V] = svd(X1)
%}

X1 = [2 1 3; 1 2 1];
X2 = [2 1; 1 2; 3 1];
X3 = [1 3 2 5; 2 5 3 9; 2 1 -1 5; 3 2 -1 8; 1 1 0 3];

rankX1 = rank(X1);
rankX2 = rank(X2);
rankX3 = rank(X3);

[U1, S1, V1] = svd(X1);
[U2, S2, V2] = svd(X2);
[U3, S3, V3] = svd(X3);


[U12, S12, V12] = svd(X1*X2);
[U21, S21, V21] = svd(X2*X1);
[U33t, S33t, V33t] = svd(X3*X3');
[U3t3, S3t3, V3t3] = svd(X3'*X3);

X1_inv = V1*pinv(S1)*U1';
X2_inv = U1*pinv(S1')*V1';
X1X2_inv = U1*pinv(S1*S1')*U1';
X2X1_inv = V1*pinv(S1'*S1)*V1';
X3_inv = V3*pinv(S3)*(U3');