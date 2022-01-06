import numpy as np
import math
from scipy.spatial import distance
from random import randint 

from typing import Dict, Union, List
import warnings  #Just to supress some RuntimeWarning - Mean of empty slice.


def random_sample(arr: np.array, n: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=n, replace=False)]


def dist(X,Y):
    return distance.euclidean(X,Y)
 

def K_Means(X,K,mu):
    X=np.array(X)
    mu=np.array(mu)
        
    warnings.filterwarnings("ignore")
    
    if type(X[0])==np.ndarray:
        no_of_features=len(X[0])
    else:
        no_of_features=1
    
    if len(mu)==0:
        mu=random_sample(X,K)
        
    iter1=0
    shape_of_mu=np.shape(mu)
    while 1:
        distances_of_each_x_point_to_mus = [[dist(each_mu,each_x_train) for each_mu in mu] for each_x_train in X]
        cluster_with_respect_to_point = [np.argmin(distances_with_mu) for distances_with_mu in distances_of_each_x_point_to_mus]
        new_mu=np.empty([0,no_of_features])
        iter1+=1
       
        for mu_position in np.arange(0,len(mu)):
            if no_of_features==1:
                new_mu=np.append(new_mu,np.mean(X[np.where(cluster_with_respect_to_point==mu_position)]))
            else:
                new_mu=np.append(new_mu,[np.mean(X[np.where(cluster_with_respect_to_point==mu_position)],axis=0)],axis=0)
        
        for i in np.argwhere(np.isnan(new_mu)):
            new_mu[i]=mu[i]
        
        if np.array_equal(mu,new_mu):
            try:
                return set(tuple(i) for i in mu)
            except:
                return set(mu)
        else:
            mu=new_mu
            
            
        

def K_Means_better(X,K):
    
    if type(X[0])==np.ndarray:
        no_of_features=len(X[0])
    else:
        no_of_features=1
     
    mus=K_Means(X,K,[])
    to_count=[[mus,1]]
    
    is_majority=False
    max_loop=10000
    loop_counter=0
    min_loop=min(len(X),10000)
        
    while not is_majority:
        loop_counter+=1
        mus=K_Means(X,K,[])
        found=False
        for j in range(0,len(to_count)):
            if mus==to_count[j][0]:
                to_count[j][1]+=1
                found=True
                break
        if found==False:
            to_count.append([mus,1])
        maximum=max([[a[1],a[0]] for a in to_count])
        is_majority= (((maximum[0] >= .5*(len(X)+1+loop_counter)) or max_loop<=loop_counter) and loop_counter>=min_loop)
        if is_majority:
            return maximum[1]
        
        



