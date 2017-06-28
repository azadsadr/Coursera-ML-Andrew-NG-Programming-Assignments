function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
	%GRADIENTDESCENT Performs gradient descent to learn theta
	%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
	%   taking num_iters gradient steps with learning rate alpha

	% Initialize some useful values
	m = length(y); % number of training examples
	J_history = zeros(num_iters, 1);

	for iter = 1:num_iters
		
		% ====================== YOUR CODE HERE ======================
		% Instructions: Perform a single gradient step on the parameter vector
		%               theta. 
		%
		% Hint: While debugging, it can be useful to print out the values
		%       of the cost function (computeCost) and gradient here.
		%
		
		% taking 1st column of X i.e. all values of feature x0 or all ones.
		x0_is = X(:, 1);
		
		% to calculate delta2 x0_is' is a 1 X m matrix and (X * theta) - y) is a m X 1 matrix
		delta1 = alpha * (1 / m) * (x0_is' * ((X * theta) - y));
		
		% taking 2nd column of X i.e. all values of feature x1.
		x1_is = X(:, 2);
		
		% to calculate delta2 x1_is' is a 1 X m matrix and (X * theta) - y) is a m X 1 matrix
		delta2 = alpha * (1 / m) * (x1_is' * ((X * theta) - y));
		
		% temporary variables for simultaneous updates.
		temp1 = theta(1) - delta1;
		temp2 = theta(2) - delta2;
		
		theta(1) = temp1;
		theta(2) = temp2;
		% ============================================================
		
		% Save the cost J in every iteration    
		J_history(iter) = computeCost(X, y, theta);
	end

end
