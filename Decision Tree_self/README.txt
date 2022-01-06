Decision Trees:
A. DT_train_binary(X,Y,max_depth):

Inputs:
X->numpy nd array of binary training data 
Y->numpy 1d array of binary training labels
max_depth->maximum depth for the tree.


Output:
DT model, which is a recursive dictionary of list and dictionary in following structure:
# #             {"X_po" 2    
# #               "div" [{"X_po" None                    
# #                       "div" 1}                       
                  
# #                      {"X_po" 0                   
# #                              "div" [{"X_po" None    
# #                                      "div" 1}                                          
# #                                     {"X_po" None}   
# #                                      "div" 0]}                                 
# #                             {"X_po" 1                
# #                              "div" [{"X_po" None     
# #                                      "div" 1}                                          
# #                                     {"X_po" None}   
# #                            ]}
# #                     ]
##                        }


In the structure above: 
        X_po: {  n  0 to N-1 if there is branching in the nth feature 
              {  None if at leaf node
        
        N: Number of features
        div:  {branches if the branching is to be done
              {label if we are at the leaf node

Note: The dictionary in index 0 of the list in “div” represents the side of the branch where the feature X[x_po] is 0 and the index 1 represents where X[x_po] is 1.
Note2: What exactly is represented by the example above is described in the code inside a comment section.


Workings:

* DT_train_binary is used to create a decision tree of binary input features with binary labels. It uses a recursive function ‘create_branch’ to create the necessary branches of the tree. The create_branch function takes following when initiated 
        X, in transpose order to that of input i.e. in order [[all x0][all x1]...[all xN]]; 
        Y, [y0,y1 …. yN]
        x_remaining, list of features remaining we can branch in
        No_of_features, total no of features
        max_depth , maximum features allowed for the tree.
* The create_branch checks if we reached the leaf node or not using the function condition_to_branch. Here, there is two possibility: 
        A. If we are not at leaf node, the function finds the feature to branch on which gives the maximum information gain and branches there and enters into a recursion. The inner recursion is provided with new X,Y and X_remaining that comes after branching, but since no_of_features and max_depth are globally same for a given  training they are the same as that of upper recursion. X and Y are updated as per the comparison of X[X_po] with “div_value” and X_remaining is updated by removing the x_po value from the previous X_remaining per recursion.
        B. If it is a leaf node, then the function returns None in “X_po” representing the leaf node and the lebel to classify in  “div”.
* The  branching condition is checked by condition_to_branch which takes the input:
        X, in transpose order to that of input i.e. in order [[all x0][all x1]...[all xN]]; 
        Y, [y0,y1 …. yN]
        x_remaining, list of features remaining we can branch in
        No_of_features, total no of features
        max_depth , maximum features allowed for the tree.
    The condition_to_branch function returns false if information gain is zero or maximum_depth is reached. It returns Information gain for provided X and Y. This way we can find which feature gives the best information gain and branch on that.
* By using the definition of Information gain and entropy, we created a function for each of them.


Possible Improvement:
1. We can create each branch as an object and do recursion in that. It makes the program shorter and more efficient.
2. We can modify the function to incorporate the creation of real discrete values.


B. DT_make_prediction(x,DT):

Inputs:
X->numpy 1d array of a single test sample
DT-> A trained decision tree 

Output:
1/0 as per the classification done.

Workings:
We pass the single numpy array to the Decision Tree and run through it to get the classification.
* If the “X_po” value of the DT is None, we just return the value of “div” as the label of the input X. 
* If the “X_po” value is some integer, we check if the X[“X_po”] is 0 or not :
        1. If X[“X_po”]==0, we move to the “div”[0] and iterate the process.
        2. If X[“X_po”]!=0, we move to the “div”[1] and iterate the process.

Possible Improvement:
1. Instead of considering the value of the index of the “div” to move forward; it is better to create a variable in the data structure of the tree to define it. This way, confusion is cleared out.


C. DT_test_binary(X,Y,DT):

Inputs:
        X->numpy nd array of a multiple test sample
        Y-> labels of the test samples
        DT-> A trained decision tree 

Output:
Accuracy of the trained binary decision tree for given testing data.

Workings:
Prediction for each test sample is done iterating the function DT_make_prediction function over X. This way we get an array of predicted labels. Then we calculate the accuracy as:
Accuracy= Number of correct prediction/No of test data*100
        where, 
              Number of correct prediction  =sum([predicted label == test labels])






D. DT_train_real(X,Y,max_depth):

Inputs:
X->numpy nd array of discrete training data 
Y->numpy 1d array of binary training labels
max_depth->maximum depth for the tree.

Output:
DT model, which is a recursive dictionary of list and dictionary. An example decision tree is :
# #             [{"X_po" 2                                    
# #               "div_val": x_po_branch_val,                 
# #               "div" [{"X_po" None                         
# #                       "div" 1}                           
                  
# #                      {"X_po" 0                            
# #                       "div_val": x_po_branch_val,        
# #                      "div" [{"X_po" 1                    
# #                              "div_val": x_po_branch_val, 
# #                              "div" [{"X_po" None         
# #                                      "div" 1}             
                                  
# #                                     {"X_po" None}         
# #                                      "div" 0]}           
                          
# #                             {"X_po" 1                     
# #                              "div_val": x_po_branch_val,  
# #                              "div" [{"X_po" None         
# #                                      "div" 1}             
                                  
# #                                     {"X_po" None}        
# #                                      "div" 0]}            
# #                            ]}
# #                     ]}
# #             ]


In the structure above: 
        X_po: {n  0 to N-1 if there is branching in the nth feature 
                {None if at leaf node      
        N: Number of features
        div_val: {(X[X_po]  if branching happens as per x>p}
                {None if leaf node}
        div: {branches if the branching is to be done}
             {label if we are at the leaf node} 
       
Note: The dictionary in index 0 of the list in “div” represents the side of the branch where the feature X[x_po] <= div_val and the index 1 represents where X[x_po] >div_val.
Note2: What exactly is represented by the example above is described in the code inside a comment section.


Workings:
DT_train_real is used to create a decision tree of discrete input features with binary labels. It uses a recursive function ‘create_branch_real’ to create the necessary branches of the tree. The create_branch_real function takes following when initiated 
X, in transpose order to that of input i.e. in order [[all x0][all x1]...[all xN]]; 
Y, [y0,y1 …. yN]
max_depth , maximum features allowed for the tree.
depth: current depth of the tree while branching
The create_branch checks if we reached the leaf node or not using the function condition_to_branch_real_data. Here, there is two possibility: 
        A. If we are not at leaf node, the function finds the feature to branch on which gives the maximum information gain and branches there and enters into a recursion. The inner recursion is provided with new X,Y and depth that comes after branching, but since max_depth are globally same for a given training they are the same as that of upper recursion. X and Y are updated as per the comparison of X[X_po] with “div_value” and depth increases per recursion.
        B. If it is a leaf node, then the function returns None in “X_po” and “div_value” representing the leaf node and the lebel to classify is stored in  “div”.
The  branching condition is checked by condition_to_branch_real_data  which takes the input:
                X, in transpose order to that of input i.e. in order [[all x0][all x1]...[all xN]]; 
        Y, [y0,y1 …. yN]
        max_depth , maximum features allowed for the tree
        depth ,the depth of the present node.
It returns false if maximum information gain is zero or depth>=maximum_depth is reached. It returns the x_po to bench on and the value of x_po to branch on for maximum information gain if further branching is to be done.
By using the definition of Information gain and entropy, we created a function for each of them.


Possible Improvement:
1. We can create each branch as an object and do recursion in that. It makes the program shorter and more efficient.
2. We can modify the function to incorporate the creation of real discrete values.


E. DT_test_real(X,Y,DT):


Inputs:
        X->numpy nd array of a multiple test sample with discrete data points
        Y-> labels of the test samples
        DT-> A trained decision tree 

Output:
Accuracy of the trained binary decision tree for given testing data.

Workings:
A prediction function DT_make_prediction_real is created that takes a single test sample with discrete data points,X and trained Decision Tree,DT  and returns the prediction. We pass the single numpy array to the Decision Tree and run through it to get the classification.
        * If the “X_po” or “div_val” value of the node is None, we just return the value of “div” as the label of the input X. 
        * If the “X_po” value is some integer, we compare X[“X_po”] with div_val :
1. If X[“X_po”]<=div_val, we move to the “div”[0] and iterate the process.
2. If X[“X_po”]>div_val, we move to the “div”[1] and iterate the process.


Prediction for each test sample is done iterating the function DT_make_prediction_real function over X. This way we get an array of predicted labels. Then we calculate the accuracy as:
Accuracy= Number of correct prediction/No of test data*100
        where, 
                Number of correct prediction=sum([predicted label == test labels])





##########################################################################################################################################################
Nearest Neighbors:

A. KNN_test(X_train,Y_train,X_test,Y_test,K):

Inputs:
X_train=>training data
Y_train=>training label
X_test=>testing data
Y_test=> testing label 
K=> the hyperparameter that limits to use of fixed neighbours to classify

Output:
Accuracy of the model developed in the testing data set.

Workings:
* Find the euclidean distance of each testing point with all the training data and select the K nearest neighbours.
* The most frequent label is selected to be the label of the point
* If there is conflict in the label selection, -1 is selected and if the distance is the same with two training points, the one that comes first in the entry of data is selected.


B. choose_K(X_train,Y_train,X_val,Y_val)

Inputs:
X_train=>training data
Y_train=>training label
X_val=>validation data
Y_val=> validation label 

Output:
The best hyperparameter K for a given set of training and validation data sets.


Workings:
* KNN_test is iterated over all possible K from 1 to length of the training data set (Practically it is seen that the k value is best among 3,4,7 or 9 but since the data provided to test run the functions is smaller and assuming such case I ran it through all possible K value). This gives accuracy of each possible K value in given training data.
* Then, the highest accuracy is found and the index+1 value of the highest accuracy is used as the best value of K. If two k have the same highest accuracy, the one with lower value is selected.


#################################################################################################################################################################
Clustering:

A.K_Means(X,K,mu):

Inputs:
X=>training data
K=>the hyperparameter to select the number of clusters 
mu=> the initial cluster centers provided.
Note: If mu=[] then K points among X is selected randomly among the given training data set

Output:
The K cluster centers 

Workings:
1. First, we check the size of mu and if size of mu != K,  K points among X is selected randomly among the given training data set as mu.
2. Then, the distance between each point and cluster is found and clusters are created.
3. The  average of cluster points is found and stored as the centroid in mus.
4. The new mu is compared with the old mu and the following case is viewed:
   * If last_mu=new_mu: return mu in set format as output of the function 
   * If last_mu!=new_mu: goto step 2 and rerun the process


B. K_Means_better(X,K):

Inputs:
X=>training data
K=>the hyperparameter to select the number of clusters 


Output:
The K cluster centers 

Workings:
1. Run K_Means with unknown centroid i.e. as mus=[], for the minimum of the length of 50% training data to be on the safe side and store the centroid values in a list of unique centroids and number of times it is repeated.
2. If the number of appearances of a centroid is in majority of the times K-Means is run, we return the centroid as output. Else, we run K-Means multiple times unless the condition of majority is achieved or run the function for some limited times and return the maximum repeated centroid.
