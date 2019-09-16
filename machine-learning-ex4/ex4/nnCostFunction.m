function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Convert y 5000x1 to a y 5000x10 with a 0-1 interval
% 1- Create a Indentity model matrix 10x10
% 2- Use the value of every y as a index to search in the 
%    identity matrix and assign the value one in that position
n = num_labels;
identityMatrix = eye(n);
Y = zeros(m,n);
for i=1:m
  Y(i,:) = identityMatrix(y(i),:); % (5000,10)
endfor

% Forward propagation:
a1 = [ones(m,1) X]; % (5000,401)

z2 = a1 *  Theta1'; % (5000,25)
a2 = [ones(size(z2, 1), 1) sigmoid(z2)]; % (5000,26)

z3 = a2 * Theta2'; % (5000x26) * (26x10) = (5000x10)
a3 = sigmoid(z3); % (5000x10)

h = a3;

% Regularization: 
% 1- Square every single theta excluding the Theta0 (.^2)
% 2- Sum axis 2, this will sum the columns.
% 3- Sum standard (rows), to get a scalar.
sumSquaredThetas1 = sum(sum(Theta1(:, 2:end).^2, 2));
sumSquaredThetas2 = sum(sum(Theta2(:, 2:end).^2, 2));
regularization = lambda * (sumSquaredThetas1 + sumSquaredThetas2) / (2 * m);

% Implement Cost Function:
% 1- Need to multiply each y_k vector [1 0 0 ... 0] to the log(h), so need to use wise product.
% 2- Need to sum by column and then sum the columns to obtain the cost.
J = (1/m) * sum(sum(-Y .* log(h) - (1-Y) .* log(1-h), 2)) + regularization;

% Calculate deltas: There's no delta 1.
d3 = a3 .- Y; % (5000 x 10) - (5000 x 10) = (5000x10)
d2 = (d3 * Theta2) .* sigmoidGradient([ones(size(z2, 1), 1) z2]); % (5000x25)
d2 = d2(:, 2:end);

% Calculate the accumulated gradients:
acc_grad2 = d3' * a2; % (10x5000) * (5000x26) = (10x26)
acc_grad1 = d2' * a1; % (25x5000) * (5000x401) = (20x401)

% Calculate regularization:
% NOTE: regularization is never calculated over Theta_zero
Theta1_reg = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_reg = [zeros(size(Theta2 ,1), 1) Theta2(:, 2:end)];
reg1 = (lambda/m) * Theta1_reg;
reg2 = (lambda/m) * Theta2_reg;

% Obtain unregularized gradient for the neural network cost function:
Theta1_grad = acc_grad1 ./ m + reg1;
Theta2_grad = acc_grad2 ./ m + reg2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
