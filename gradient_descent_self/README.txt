Perceptron:
 
perceptron_train(X,Y):
Inputs: X- an array of training feature set
    Y- an array training labels
Output: [weights,bias]
    where, weights - the trained weights of n-dimension as of feature
           bias - the bias on the activation of perceptron
For each iteration step, we multiply the feature set of a training data to the weight and add the bias to get activation. If activation*label for the training set is greater than 0, we don't update anything else we update the weight and bias as:
     Weight <- weight + label*feature_set
     bias <- bias + label
We conduct this step for the whole dataset, i.e for an epoch. If at the end of the epoch the weight and bias is as the initial weight and bias of the epoch, we return the weight and bias as the model. If not, we iterate until the above mentioned condition is fulfilled or till some maximum number of epochs is reached (here 5000 is taken as maximum number of epochs).

perceptron_test(X_test,Y_test,w,b):
Inputs: X_test- an array of testing feature set
    Y_test- an array of testing labels
    w- trained weight of the each feature for the perceptron model 
    b- bias for the perceptron model 
Output: accuracy in range from 0-1
Each test feature set is passed to a function along with the weight and bias of the perceptron that gives out the prediction. The prediction is compared with the label and correct prediction counted. Finally, the correct prediction is divided by the total training data and the result is returned as accuracy.

Gradient Descent:

Gradinet_descent (gradinent_fn,initial_X,eta):
Inputs: gradinet_fn- the gradient of the function where gradient descent is to be implemented
    initial_X- the initial points to start gradient descent
    eta- the step size of gradient descent to be used
Outputs: the local minima as per the gradient descent

The gradient at initial point is found and the points are updated as the formula:
    X<- X-eta*gradient_at_X
Then the absolute value of gradient is checked to be either below some fixed small value or not so to test convergence limit. If the convergence is not obtained for some large number of iterations (here we take it to be 10000), we end the loop and give the result of the last iteration. This occurs if the step size is very large resulting in overshooting. The condition of very high number of iterations may also occur when the rate of descent is very small and so cannot reach to the local minima within the steps. 

