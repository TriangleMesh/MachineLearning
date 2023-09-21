function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%%%%%%%%%%% conditions %%%%%%%%%%%%
% theta = [1 ; 1];
% J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
% size of X: 12*2
% size of y: 12*1
% size of theta: 2*1

%%%%%%%%%%% J %%%%%%%%%%%%
theta_without0 = theta(2:end,:);
h = X *theta;
J  = (1/2/m)*sum((h-y).^2)+(lambda/2/m) * sum(theta_without0.^ 2);

%%%%%%%%%%% grad %%%%%%%%%%%%

theta0equals0 = [zeros(1,1);theta_without0];
grad = (1/m)*X'*(h-y) + lambda/m * theta0equals0;






% =========================================================================

grad = grad(:);

end
