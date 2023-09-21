function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = (theta' * X')';
g = sigmoid(z);
c = -y .* log(g);
d = (1 .- y) .* log(1 .- g);
e = c-d;

%costFunction(theta, X, y);
B = theta(2:end,1);

A = lambda*sum(B .^ 2)/(2*m);
C = [0; B];

J = sum(e)./m + A;
grad = ((X' * (g-y)) ./ m )+ (C*lambda/m);



% =============================================================

end
