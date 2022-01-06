import numpy as np
import math
from scipy.spatial import distance
from random import randint 

from typing import Dict, Union, List

def maximum_freq_np(np_array):
    values,counts = np.unique(np_array,return_counts=True)
    return(values[np.argmax(counts)])

def predict_KNN(X_train,Y_train,X_test_point,K):
    distances_with_training_X= [distance.euclidean(X_test_point,train_xes) for train_xes in X_train]
    smallest_k_distant_indexes=np.argpartition(distances_with_training_X,K)[:K]
    
    ys_of_smallest_distances=[Y_train[i] for i in smallest_k_distant_indexes]
    prediction_class=maximum_freq_np(ys_of_smallest_distances)
    return prediction_class

def find_accuracy(actual_Y,predicted_y):
    actual_Y=np.array(actual_Y)
    predicted_y=np.array(predicted_y)

    no_of_correct_prediction=np.sum(actual_Y==predicted_y)
    total_test_data_count=len(actual_Y)
    accuracy=(no_of_correct_prediction/total_test_data_count)*100
    return accuracy

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    X_test=np.array(X_test)
    Y_test=np.array(Y_test)
    
    predicted_ys=[predict_KNN(X_train,Y_train,X_test_points,K) for X_test_points in X_test]
    accuracy=find_accuracy(Y_test,predicted_ys)
    return accuracy

def choose_K(X_train,Y_train,X_val,Y_val):
    accuracies_in_various_k=np.array([KNN_test(X_train,Y_train,X_val,Y_val,K) for K in range(1,len(Y_train))])
    best_k = np.argmax(np.array([accuracies_in_various_k]))+1
    return best_k