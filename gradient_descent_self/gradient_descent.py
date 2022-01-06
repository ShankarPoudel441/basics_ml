import numpy as np
import copy


def gradient_descent(gradient_fn,initial_X,eta):
    total_count=0
    X=np.array(initial_X)
    
    while 1:
        gradient=np.array(gradient_fn(X))
        new_X=X-eta*gradient
        converged = len(gradient[np.abs(gradient)>0.0001])<1
        if converged or total_count>=10000:
            if total_count>=10000:
                print("Not converged due to high value of step size eta or the rate of convergence is exceptionally small for the problem")
            break
        total_count-=1
        X=copy.deepcopy(new_X)
    return X

        