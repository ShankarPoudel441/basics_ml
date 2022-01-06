import numpy as np
import math
from random import randint 

from typing import Dict, Union, List

def Entropy(probabilities: List[float]):
#     Find entropy given the probabilities
    return_entropy = 0
    for probability_i in probabilities:
        if probability_i>0:
            return_entropy-=probability_i*math.log2(probability_i)
    return return_entropy


def Information_Gain(total_y: List[int],
                     y_when_x_equals_yes: List[int],
                     y_when_x_equals_no: List[int]):
    
# #     Find Information Gain from the condition where the Y values that is divided as necessary in Yes and No...
# #     ... conditions of X. Here all the inputs are the Y values that is divided into two sets as per the ...
# #     ... values of x to be looked upon 

    total_y=np.array(total_y)
    y_when_x_equals_yes=np.array(y_when_x_equals_yes)
    y_when_x_equals_no=np.array(y_when_x_equals_no)
    
    N_Total_1= np.count_nonzero(total_y==1)
    N_Total_0= np.count_nonzero(total_y==0)
    N_Total= len(total_y)
    N_x_equals_yes_1=np.count_nonzero(y_when_x_equals_yes==1)
    N_x_equals_yes_0=np.count_nonzero(y_when_x_equals_yes==0)
    N_x_equals_yes=len(y_when_x_equals_yes)
    N_x_equals_no_1=np.count_nonzero(y_when_x_equals_no==1)
    N_x_equals_no_0=np.count_nonzero(y_when_x_equals_no==0)
    N_x_equals_no=len(y_when_x_equals_no)
    
    if (N_Total==0 or N_Total_1==0 or N_Total_0==0):
        return 0
    
    Entropy_Total = Entropy([N_Total_1/N_Total,N_Total_0/N_Total])
    Entropy_x_equals_yes = 0 if (N_x_equals_yes_1==0 or N_x_equals_yes_0==0) else Entropy([N_x_equals_yes_1/N_x_equals_yes,N_x_equals_yes_0/N_x_equals_yes])
    Entropy_x_equals_no = 0 if (N_x_equals_no_1==0 or N_x_equals_no_0==0) else Entropy([N_x_equals_no_1/N_x_equals_no,N_x_equals_no_0/N_x_equals_no])
    
    Information_Gain=Entropy_Total-(N_x_equals_yes/N_Total*Entropy_x_equals_yes)-(N_x_equals_no/N_Total*Entropy_x_equals_no)
    return Information_Gain


def find_ig_given_X_Y(single_X,Y):     
#Eg:X0=[1,1,1,0],Y=[1,0,0,1]   Yes/No for X0 and One/Zero for Y
    X_equals_yes = np.array(Y[single_X==1]).flatten()
    X_equals_no= np.array(Y[single_X==0]).flatten()
    return Information_Gain(Y,X_equals_yes,X_equals_no)


def ig_of_all_provided(X_in_order,Y):
    IG_of_all_given=[]
    for x in X_in_order:
        ig=find_ig_given_X_Y(x,Y)
        IG_of_all_given.append(ig)
    return IG_of_all_given


def actual_X_to_branch_on(IG_of_all_given,remaining_x):
    index_of_max_ig=np.argmax(IG_of_all_given)
    return remaining_x[index_of_max_ig]


def maximum_freq_np(np_array):
    values,counts = np.unique(np_array,return_counts=True)
    return(values[np.argmax(counts)])


def condition_to_branch(X,Y,x_remaining,no_of_features,max_depth):
    if max_depth <= no_of_features-len(x_remaining) or len(x_remaining)<=0:
        return False
    IG_of_all=ig_of_all_provided(X,Y)
    if max(IG_of_all) == 0:
        return False
    return IG_of_all



def binary_branching_removing_the_feature_to_branch_on(X,Y,x_remaining):
#     Inputs: X ==> The features in structure of [[all X0] [all x1] [allX2]]... (Trasverse of input designed)
#         Y  ==> (the classes of each training set), Here,(len(X[0]) = len(Y))
#         index_to_branch_on ==> the index in given X to branch upon
#     Outputs: [{"X": x0,"Y":y0},{"X":x1, "Y":y1}] 
#         where; the first dict inside the list is for branch with X_to_branch_on = 0 
#         and the second dict inside the list is for branch with X_to_branch_on = 1
        
#     This function returns the X,Y for each branch to be branched on the index value index_to_branch_on 
#     the index_to_branch_on is the temporary index that we get among the passed X value; 
#     the index_to_branch_on is not derived from the the remaining_X values
    
    IG_of_all=ig_of_all_provided(X,Y)
    index_to_branch_on =np.argmax(IG_of_all)
    feature_to_branch_on = x_remaining[index_to_branch_on]
    x_remaining = x_remaining[x_remaining!=feature_to_branch_on]
    
    
    y1 = Y[X[index_to_branch_on]>0]
    y0 = Y[X[index_to_branch_on]<=0]
    x0=[]
    x1=[]
    for xes_in_given_X in range(len(X)):
        if xes_in_given_X!=index_to_branch_on:
            x0.append(X[xes_in_given_X][X[index_to_branch_on]<=0])
            x1.append(X[xes_in_given_X][X[index_to_branch_on]>0])
    return [{"X": x0,"Y":y0,"x_remaining":np.array(x_remaining).flatten()},
            {"X":x1, "Y":y1, "x_remaining":np.array(x_remaining).flatten()}]
    
    
    
    
def create_branch(X,Y,x_remaining,no_of_features,max_depth):
    condition = condition_to_branch(X,Y,x_remaining,no_of_features,max_depth)
    if condition:
        index_of_max_ig=np.argmax(condition)
        feature_to_branch_on = x_remaining[index_of_max_ig]
        branching_step = binary_branching_removing_the_feature_to_branch_on(X,Y,x_remaining)
        return {"X_po": feature_to_branch_on,
                 "div": [create_branch(branching_step[0]["X"],branching_step[0]["Y"],branching_step[0]["x_remaining"],no_of_features,max_depth),
                         create_branch(branching_step[1]["X"],branching_step[1]["Y"],branching_step[1]["x_remaining"],no_of_features,max_depth)]}
    else:
        return {"X_po": None,
                "div": maximum_freq_np(Y)}
        
        
        
def DT_train_binary(X,Y,max_depth):
# Input: X=[[0X0,0X1,0X2...][1X0,1X1,1X2,...],...]
#        Y=[1Y,1Y,2Y,...]


# #     Outout:
# #             [{"X_po" 2                                ==>> Branch in position 2
# #               "div" [{"X_po" None                     ==> the left-node i.e x2=0 node:: here is no branching as "X_po"== None
# #                       "div" 1}                        ==> the non branching node i.e. leaf node here is giving the class of 1 i.e if x2==0 -> Y= 1
                    
# #                      {"X_po" 0                        ==> the right-node from parent node i.e. x2=1 node:: here is again branching in X0 
# #                      "div" [{"X_po" 1                 ==> the left-node from parent node i.e. x0=0:given x2=1 is again branching in X1
# #                              "div" [{"X_po" None      ==> the left-node from parent node i.e. x1=0: given x0=0 given x2=1 branch:: here is no branching as "X_po"== None
# #                                      "div" 1}         ==> the above mentioned leaf node is classified into class 1 
                                    
# #                                     {"X_po" None}     ==> the right-node from parent node i.e. x1=1: given x0=0 given x2=1 branch:: here is no branching as "X_po"== None
# #                                      "div" 0]}        ==> the above mentioned leaf node is classified into class 1
                            
# #                             {"X_po" 1                 ==> the right-node from parent nodei.e. x0=1:given x2=1 is again branching in X1
# #                              "div" [{"X_po" None      ==> the left-node i.e. x1=0: given x0=1 given x2=1 branch:: here is no branching as "X_po"== None
# #                                      "div" 1}         ==> the above mentioned leaf node is classified into class 1
                                    
# #                                     {"X_po" None}     ==> the right-node i.e. x1=1: given x0=1 given x2=1 branch:: here is no branching as "X_po"== None
# #                                      "div" 0]}        ==> the above mentioned leaf node is classified into class 1
# #                            ]}
# #                     ]}
# #             ]
      

    no_of_features = len(X[0])
    max_depth = min(no_of_features, no_of_features if (max_depth==-1) else max_depth)
    
    initial_remaining_x = np.arange(no_of_features)
    X=np.array(X).T    #The X calculated and used in the Trasverse structure to the defined input structure i.e. in structure X=[[X0 of all data][X1 of all data]..]
    Y=np.array(Y)
    DT=create_branch(X,Y,initial_remaining_x,no_of_features,max_depth)
    return DT




def DT_make_prediction(x,DT):
    X_po = DT["X_po"]
    div=DT["div"]
    while X_po!=None:
        if x[X_po]<=0:
            X_po=div[0]["X_po"]
            div=div[0]["div"]
        else:
            X_po=div[1]["X_po"]
            div=div[1]["div"]
    return div

def DT_test_binary(X,Y,DT):
    prediction_y=np.array([DT_make_prediction(x_instance,DT) for x_instance in X])
    no_of_correct_prediction=np.sum(Y==prediction_y)
    total_test_data_count=len(Y)
    accuracy=(no_of_correct_prediction/total_test_data_count)*100
    return accuracy


# # Defining main function
# def main():
#     X=[[0,1,0,1],[1,1,1,1],[0,0,0,1],[0,1,0,0]]
#     Y=[1,1,0,0]
#     model_data2=DT_train_binary(X,Y,3)
#     # DT_test_binary(X,Y,model_data2)
  
  
# # Using the special variable 
# # __name__
# if __name__=="__main__":
#     main()
    
    
    

##########  Real Values DT Creations############
################################################
# Here it starts exactly for all the real valued functions 
# We can easily change/modify majority of functions of binary valued input to make useful for both conditions too but I beacme lazy 
# S0, instead of modifying them, I created new ones and completed the tasks :)
# Also, I could have uesd the real valued functions itself to solve the binary valued questions, as binary value question is subset of 
# real valued one but it is seen that what I did for binary valued problem is more efficient timewise; hence I let it be as it is. 
################################################
################################################

    
def find_Y_above_and_below(x0,Y,option):
    x0=np.array(x0)
    Y=np.array(Y)
    
    Y_when_s_eq_below=Y[x0<=option]
    Y_when_s_above=Y[x0>option]
    return [Y_when_s_above,Y_when_s_eq_below]

def find_ig_given_x_options(x,Y,option):
    x0=np.array(x)
    Y=np.array(Y)
    
    aboves,belows=find_Y_above_and_below(x,Y,option)
    ig=Information_Gain(Y,aboves,belows)
    return ig

def find_branching_options(x):
    x=np.array(x)
    return np.unique(x)

def find_ig_given_x(x,Y):
    branching_options_101 = find_branching_options(x)
    igs=[find_ig_given_x_options(x,Y,option) for option in branching_options_101]
    branching_value_for_max_ig=branching_options_101[np.argmax(igs)]
    max_ig=np.max(igs)
    return [branching_value_for_max_ig,max_ig]

def branch_where_what(X,Y):
    branches_possible=np.array([find_ig_given_x(x,Y) for x in X])
    max_IG=np.max(branches_possible.T[1])
    x_po=np.argmax(branches_possible.T[1])
    x_po_branch_val=branches_possible.T[0][x_po]
    return(x_po,x_po_branch_val,max_IG)

def condition_to_branch_real_data(X,Y,max_depth,depth):
    if max_depth<=depth:
        return False
    x_po,x_po_branch_val,max_IG = branch_where_what(X,Y)
    if max_IG==0:
        return False
    else:
        return([x_po,x_po_branch_val])  
    
    
def binary_branching_real_data(X,Y,x_po,x_po_branch_val):
    x_for_x_greter_than_val=[x[X[x_po]>x_po_branch_val] for x in X]
    y_for_x_greter_than_val=Y[X[x_po]>x_po_branch_val]
    x_for_x_less_eq_than_val=[x[X[x_po]<=x_po_branch_val] for x in X]
    y_for_x_less_eq_than_val=Y[X[x_po]<=x_po_branch_val]
    
    
    return [{"X": x_for_x_less_eq_than_val,
             "Y": y_for_x_less_eq_than_val},
            {"X": x_for_x_greter_than_val,
              "Y": y_for_x_greter_than_val}]
 
 
def maximum_freq_np(np_array):
    values,counts = np.unique(np_array,return_counts=True)
    return(values[np.argmax(counts)])


def create_branch_real(X,Y,max_depth,depth):
    condition = condition_to_branch_real_data(X,Y,max_depth,depth)
    if condition:
        x_po=condition[0]
        x_po_branch_val = condition[1]
        
        branching_step = binary_branching_real_data(X,Y,x_po,x_po_branch_val)
        
        return {"X_po": x_po,
                "div_val": x_po_branch_val,
                 "div": [create_branch_real(branching_step[0]["X"],
                                       branching_step[0]["Y"],
                                       max_depth,
                                       depth+1),
                         create_branch_real(branching_step[1]["X"],
                                       branching_step[1]["Y"],
                                       max_depth,
                                       depth+1)]}
    else:
        return {"X_po": None,
                "div_val": None,
                "div": maximum_freq_np(Y)}

# def flatten_array_object(array_object,final_return_val=[]):
#     if type(array_object[0])==int or type(array_object[0])==float or type(array_object[0])==str or type(array_object[0])==np.float64:
#         final_return_val=np.append(final_return_val,array_object[0])
#     else:
#         flatten_array_object(array_object[0],final_return_val)
#     if (len(array_object)==1 or len(array_object)==0):
#         return final_return_val
#     else:
#         flatten_array_object(array_object[1:],final_return_val)

def all_possible_branchings(X_ie_transpose_of_input):
    total_possibles= np.array([find_branching_options(x) for x in X_ie_transpose_of_input],dtype='object')
    return total_possibles
    
def total_possibile_depth(X_ie_transpose_of_input):
    total_possibles=all_possible_branchings(X_ie_transpose_of_input)
    total_depth_possible=np.sum(np.array([len(x) for x in total_possibles]))
    return total_depth_possible
  


def DT_train_real(X,Y,max_depth):
# Input: X=[[0X0,0X1,0X2...][1X0,1X1,1X2,...],...]
#        Y=[1Y,1Y,2Y,...]


# #     Outout:
# #             [{"X_po" 2                                    ==>> Branch in position 2
# #               "div_val": x_po_branch_val,                 ==>> Value with which comparison is done to divide into branch in if(x2>div_val -> got to right branch) else goto left branch
# #               "div" [{"X_po" None                         ==> the left-node i.e x2=0 node:: here is no branching as "X_po"== None
# #                       "div" 1}                            ==> the non branching node i.e. leaf node here is giving the class of 1 i.e if x2==0 -> Y= 1
                    
# #                      {"X_po" 0                            ==> the right-node from parent node i.e. x2=1 node:: here is again branching in X0 
# #                       "div_val": x_po_branch_val,         ==>> Value with which comparison is done to divide into branch in if(x0>div_val -> got to right branch) else goto left branch
# #                      "div" [{"X_po" 1                     ==> the left-node from parent node i.e. x0=0:given x2=1 is again branching in X1
# #                              "div_val": x_po_branch_val,  ==>> Value with which comparison is done to divide into branch in if(x1>div_val -> got to right branch) else goto left branch
# #                              "div" [{"X_po" None          ==> the left-node from parent node i.e. x1=0: given x0=0 given x2=1 branch:: here is no branching as "X_po"== None
# #                                      "div" 1}             ==> the above mentioned leaf node is classified into class 1 
                                    
# #                                     {"X_po" None}         ==> the right-node from parent node i.e. x1=1: given x0=0 given x2=1 branch:: here is no branching as "X_po"== None
# #                                      "div" 0]}            ==> the above mentioned leaf node is classified into class 1
                            
# #                             {"X_po" 1                     ==> the right-node from parent nodei.e. x0=1:given x2=1 is again branching in X1
# #                              "div_val": x_po_branch_val,  ==>> Value with which comparison is done to divide into branch in if(x1>div_val -> got to right branch) else goto left branch
# #                              "div" [{"X_po" None          ==> the left-node i.e. x1=0: given x0=1 given x2=1 branch:: here is no branching as "X_po"== None
# #                                      "div" 1}             ==> the above mentioned leaf node is classified into class 1
                                    
# #                                     {"X_po" None}         ==> the right-node i.e. x1=1: given x0=1 given x2=1 branch:: here is no branching as "X_po"== None
# #                                      "div" 0]}            ==> the above mentioned leaf node is classified into class 1
# #                            ]}
# #                     ]}
# #             ]
    
    
    X=np.array(X).T    #The X calculated and used in the Trasverse structure to the defined input structure i.e. in structure X=[[X0 of all data][X1 of all data]..]
    Y=np.array(Y)

    total_possibility_of_depth=total_possibile_depth(X)
    max_depth = min(total_possibility_of_depth, total_possibility_of_depth if (max_depth==-1) else max_depth)
    
    depth = 0
    
    DT=create_branch_real(X,Y,max_depth,depth)
    return DT


def DT_make_prediction_real(x,DT):
    X_po = DT["X_po"]
    div=DT["div"]
    div_val= DT["div_val"]
    while X_po!=None:
        if x[X_po]<=div_val:
            X_po=div[0]["X_po"]
            div_val= div[0]["div_val"]
            div=div[0]["div"]
        else:
            X_po=div[1]["X_po"]
            div_val= div[1]["div_val"]
            div=div[1]["div"]
    return div



def DT_test_real(X,Y,DT):
    X=np.array(X)
    Y=np.array(Y)
    prediction_y=[DT_make_prediction_real(x,DT) for x in X]
    no_of_correct_prediction=np.sum(Y==prediction_y)
    total_test_data_count=len(Y)
    accuracy=(no_of_correct_prediction/total_test_data_count)*100
    return accuracy
