import numpy as np
import sklearn as sk
import matplotlib as pl
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

def softmax_1d (X):
    exponentials=[np.exp(x) for x in X]
    total=np.sum(exponentials)
    return np.array(exponentials/total)

def predict_one_betn_0and1(model,X):
    a = [np.matmul(x,model["W1"]) for x in X] +model['b1']            #N rows of vectors od size of hiddenlayer(l) 
    H = np.tanh(a)                                                      #N rows of vectors od size of hiddenlayer(l) 
    Z=  np.array([np.dot(h,model["W2"]) for h in H] + model['b2']) 
    y_cap=np.array([softmax_1d(z) for z in Z])                                    #N rows of vectors of size 2 
    return y_cap

def predict(model,X):
    y_cap=predict_one_betn_0and1(model,X)
    y_return=[0 if y[0]>y[1] else 1 for y in y_cap]
    return np.array(y_return)

def calculate_loss(model, X, y):
    Total_no_of_X = len(X)
    y_in_2d = [[1,0] if y1==0 else [0,1] for y1 in y] 
    y_cap=predict_one_betn_0and1(model,X)
    loss=y_in_2d*np.log(y_cap)
    Loss = -np.sum(loss)/Total_no_of_X
    return Loss


def batch_back_propagation_and_updation(model,X,y_in_2d,learining_rate=0.01): 
    
    if len(X)==0 or len(y_in_2d)==0:
        return model

    #Feed forward
    a = [np.matmul(x,model["W1"]) for x in X] +model['b1']            #N rows of vectors od size of hiddenlayer(l) 
    H = np.tanh(a)                                                      #N rows of vectors od size of hiddenlayer(l) 
    Z=  np.array([np.dot(h,model["W2"]) for h in H] + model['b2'])              #N rows of vectors of size 2 
    y_cap=np.array([softmax_1d(z) for z in Z])                                    #N rows of vectors of size 2 
    
    #Derivatives
    dl_by_dycap = y_cap-y_in_2d                                        # of size ycap
    dl_by_da=(1-np.square(np.tanh(a)))*np.dot(dl_by_dycap,model['W2'].T) #of size a
    dl_by_dw2=np.matmul(H.T,dl_by_dycap)              #of size 
    dl_by_db2=dl_by_dycap                             #of size b2
    dl_by_dw1=np.matmul(X.T,dl_by_da)                 #of size w1
    dl_by_db1=dl_by_da                                #
        
    #Updation
    w1=model["W1"]-learining_rate*dl_by_dw1
    w2=model["W2"]-learining_rate*dl_by_dw2
    b1=model["b1"]-learining_rate*np.mean(dl_by_db1,0)
    b2=model["b2"]-learining_rate*np.mean(dl_by_db2,0)
    
    model={"W1":w1,
           "W2": w2,
           "b1":b1,
           "b2":b2}
    return model


def build_model(X, y, nn_hdim , num_passes =20000 , print_loss=False) :
    np.random.seed(1)
    y_in_2d = [[1,0] if y1==0 else [0,1] for y1 in y]
    model = {'W1': np.random.normal(0,1/(2*nn_hdim),[2,nn_hdim]), 'b1': np.random.normal(0,1/nn_hdim,nn_hdim),
             'W2':np.random.normal(0,1/(2*nn_hdim),[nn_hdim,2]), 'b2': np.random.normal(0,1/2,2)}
    batch_size=5
    no_of_batches=len(y)//batch_size+1
    for i in range(num_passes):
        for batch_id in range(no_of_batches):
            x_batch=X[batch_size*batch_id:batch_size*(batch_id+1)]
            y_in_2d_batch=y_in_2d[batch_size*batch_id:batch_size*(batch_id+1)]
#             print("x_batch=",x_batch)
#             print("y_batch=",y_in_2d_batch)
            model=batch_back_propagation_and_updation(model,x_batch,y_in_2d_batch,learining_rate=0.01)
        if i%1000 == 0:
            if print_loss==True:
                print("Loss=",calculate_loss(model,X,y))
    return model