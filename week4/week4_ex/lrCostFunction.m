function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% theta = [-2; -1; 1; 2];
% X = [ones(5,1) reshape(1:15,5,3)/10];
% y = ([1;0;1;0;1] >= 0.5);
% lambda_t = 3;
% [J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

% g = 1.0 ./ (1.0 + exp(-z));

% z = X * theta;
g = sigmoid( X * theta);
c = -y .* log(g);
d = (1 .- y) .* log(1 .- g);
e = c-d;
J_c = sum(e)./m;

grad = grad(:);

% grad_c = ((g-y)'* X)'./m; 
grad_c = (X' * (g-y)) ./ m ;

% theta_c = theta(2:end);
% theta_d = [0;theta_c];
temp = theta;
temp(1) = 0;

J = J_c + lambda*sum(theta(2:end) .^ 2)/(2*m);
grad = grad_c + temp * lambda / m;



% B = theta(2:end,1);

% A = lambda*sum(B .^ 2)/(2*m);
% C = [0; B];

% J = sum(e)./m + A;
% grad = ((X' * (g-y)) ./ m )+ (C*lambda/m);







% =============================================================

% grad = grad(:);
% grad = ((X * (g-y)) ./ m )+ (C*lambda/m);

end
