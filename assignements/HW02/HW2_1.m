clc
clear all
close all

A = [1 2 -3; 4 5 6];
b = [12; 0];
aug_Ab = [A b];

A_pinv = pinv(A);
A_right_inv = A'*inv(A*A');

% why not same solutions?! because in 
% uderdetermined system could have infinite solutions (if full rank)
% but it could have one unique solution by minimum norm
x = A\b; %one arbitrary solution
x_ = A_right_inv*b; % the solution that corresponds to min norm solution
x_norm = lsqminnorm(A, b);

rank(A);
rank(aug_Ab);
A*A_right_inv

% ---A_right_inv*A--- will not result in identity