import numpy as np
import copy


def perceptron_train(X,Y):
    X=np.array(X)
    Y=np.array(Y)
    
    no_of_features=len(X[0])
    w=np.zeros(no_of_features)
    b=0
    
    new_w=copy.deepcopy(w)
    new_b=copy.deepcopy(b)
    
    max_epochs=5000
    
    while 1:
        for counter, label in enumerate(Y):
            max_epochs=max_epochs-1
            activation=np.dot(new_w,X[counter])+new_b
            if activation*label <= 0:
                new_w=w+X[counter]*label
                new_b=b+label
        if (np.sum(new_w==w)==no_of_features and new_b==b) or max_epochs<=0:
            break
        else:
            w=copy.deepcopy(new_w)
            b=copy.deepcopy(new_b)
    return new_w,new_b


# def perceptron_one_test(X_test, Y_test, w, b):
#     x=np.array(X_test)
#     y=Y_test
    
#     prediction=np.dot(x,w)+b
#     if prediction > 0:
#         return 1
#     else:
#         return -1
    
def perceptron_test(X_test, Y_test, w, b):
    y_pred=[]
    for x in X_test:
        y_p=np.dot(x,w)+b
        if y_p >0:
            y_p=1
        else:
            y_p=-1
        y_pred.append(y_p)
    
    total=len(Y_test)
    correct=np.sum(Y_test==y_pred)
    accuracy=correct/total
    return accuracy 
        
