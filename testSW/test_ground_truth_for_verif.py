import numpy as np
from utils import *
import ot



dim_list= [3]
x = np.load("all_ground_truths3.npy")
L = []
N = 1000000000
print(x)
for i in range(len(x))
    if i != 1
        continue
    else:
    # ground_truth = MY_SW(X,Y,10000, d)
    # ground_truth = ot.sliced_wasserstein_distance(X,Y,n_projections = 10000)
        ground_truth = x[i]
        ground_truth*=10000
        
        # for i in range(10000):
        #     ground_truth+= pow(ot.sliced_wasserstein_distance(X,Y,n_projections = 10000),2)
        # L.append(ground_truth/(10000))
        continue
    
    # print("PASSED")
    # np.save("all_ground_truths2.npy", np.array(L))
        #print("PASSED")
