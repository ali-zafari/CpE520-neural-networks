clc
clear all
close all

A = [-11 2; 2 3; 2 -1];
b = [0; 7; 5];
aug_Ab = [A b];

A_pinv = pinv(A);
A_left_inv = inv(A'*A)*A';

% both below will result in equivalent solutions.
% overdetermined has at most one solution (if full rank)
% or no solutions at all (if not full rank)
x = A\b;
x_ = A_left_inv*b;

rank(A)
rank(aug_Ab)
A_left_inv * A

% ---A*A_right_inv--- will not result in identity