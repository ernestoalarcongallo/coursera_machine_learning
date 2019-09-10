function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


n = size(X, 2);

a1 = zeros(m,n+1);
a1(1,1) = 1;
a1(1,2:(n+1)) = X;

for j = 1:2
  z = Theta(j) * a(j)';
  g_z = sigmoid(z);
  p(j) = 
end

fprintf('m %f \n', m);
fprintf('n %f \n', n);
fprintf('Theta1 %f \n', size(Theta1));
fprintf('Theta2 %f \n', size(Theta2));
fprintf('X %f \n', size(X));
fprintf('a1 %f \n', size(a1));


% =========================================================================


end
