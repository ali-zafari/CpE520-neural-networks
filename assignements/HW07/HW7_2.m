clc
clear variables
close all

% every sample of input lies on a column vector of X
X = zeros(3, 8);
x1 = [0 1];
x2 = [0 1];
x3 = [0 1];

% Input sample generation
for i = 1:size(x1, 2)
    for j = 1:size(x2, 2)
        for k = 1:size(x3, 2)
            X(:, k+2*(j-1)+4*(i-1)) = [x1(i), x2(j), x3(k)];
        end
    end
end

%target outputs
t = [0 1 1 0 1 0 0 1];

% the initialization which converges
rng(10);
net = patternnet(3);
net.divideFcn = 'dividetrain';
net.performFcn = 'mse';
net.trainParam.showCommandLine = 1;
net.trainParam.show = 1;

[net, tr, y, e] = train(net, X, t);

weights_in = net.IW;
weights_hidden = net.LW;
biases  = net.b;
view(net);