neural_network.py contains six functions in it:
1. softmax_1d(X): 
	Input: X - a list or array
	Output: a numpy array 
	Explanation:
	Here, the functions returns the softmax of the input.
	
2. predict_one_betn_0and1(model,X)
	Input: model- a neural network model in the format model={"W1":w1, "W2": w2,"b1":b1,"b2":b2}
	       X - the numpy array of training data of shape N X 2, where N is the length of training data
	Output: a numpy array of size N X 2, where each represent the probability of being in class 0 and 1.
	Explanation:
	Here, we need a model of a neural network having two input, one hidden layer of any size and two outputs. We feed forward the training data to the model. In this project the activation in hidden layer is tanh and the activation in the output layer is softmax. Since the output layer as softmax function, we define the two outputs as the probability of the data to be in class 0 or 1. The list of this probabilities if each input data is the output.

3.predict(model,X):
	Input: model- a neural network model in the format model={"W1":w1, "W2": w2,"b1":b1,"b2":b2}
		X - the numpy array of training data of shape N X 2, where N is the length of training data
	Output: a numpy array of size N whcih is the prediction of the class for the input data.
	Explanation:
	Here, we pass the input as it is to the predict_one_betn_0and1 and get the prediction in probability format. We simply convert the probabilistic prediction to class prediction i.e we compare the probability and predict the class having higher value.
	
4. calculate_loss(model,X,y):
	Input: model- a single hidden layer neural network model in the format model={"W1":w1, "W2": w2,"b1":b1,"b2":b2}
		X - the numpy array which is a batch of training data
		y: ground truth of the input values
	Output: the categorical cross-entropy loss of the prediction by the model to the input data
	Explanation:
	It is the loss function for the neural-network we are defining. We are using categorical cross-entropy loss for the system. We get the probabilistic prediction using the predict_one_betn_0and1 function. Then we calculate the loss using the cross-entropy loss defiation.
	
5. batch_back_propagation_and_updation(model,X,y_in_2d,learining_rate=0.01)
	Input: model- a single hidden layer neural network model in the format model={"W1":w1, "W2": w2,"b1":b1,"b2":b2}
		X - the numpy array which is a batch of training data
		y_in_2d: ground truth in the format of probability of 0 and 1 i.e in 2D version
		learnng_rate- the rate of learning, default is 0.01
	Output: updated model 
	Explanation:
	Here, we implement the back propagation algorithm and update the parameters of the model using gradient descent. We do a single update for the whole batch of input training data.
	
6. build_model(X, y, nn_hdim , num_passes =20000 , print_loss=False):
	Input: X - the numpy array which is a batch of training data
		y-The ground truth of the input data
		nn_dim- the dimension of hidden layer
		num_pass- no of epoches to run for the training; default is 20000
		print_loss - option to print loss after 1000 epoches
	Output: a single hidden layered neural network model in the format model={"W1":w1, "W2": w2,"b1":b1,"b2":b2}
	Explanation:
	Here, we take the training data and build a neural net model. We used batch updation with batch size of 5 here. We can make batch size a input parameter and use that to change the program in the future for better options. 
