import numpy as np
from utils import *
import ot



dim_list= [2,3,5,10,20,50]
x = np.load("all_ground_truths2.npy")
L = []
N = 100000000
print(x)
for d in dim_list:
    X = np.load("X_"+str(d)+"D.npy")
    Y = np.load("Y_"+str(d)+"D.npy")
    if d == 2:
        N = 1000000
        points = np.zeros((N,2))
        for i in range(points.shape[0]):
            points[i][0] = np.cos(np.pi*i/N)
            points[i][1] = np.sin(np.pi*i/N)

        ground_truth = 0
        # print(range(0,N,10000))
        for i in range(1000):
            start_i = i*1000
            end_i = start_i + 1000
            if end_i > N:
                break

            ground_truth += pow(ot.sliced_wasserstein_distance(X,Y,projections = points[start_i:end_i].T),2)
        # ground_truth = MY_SW_2(X,Y, points)
        # x[0] = ground_truth/1000
        L.append(ground_truth/1000)
        # np.save("all_ground_truths3.npy", x)
        print(x)
    else:
    # ground_truth = MY_SW(X,Y,10000, d)
    # ground_truth = ot.sliced_wasserstein_distance(X,Y,n_projections = 10000)
        ground_truth = 0
        for i in range(10000):
            ground_truth+= pow(ot.sliced_wasserstein_distance(X,Y,n_projections = 10000),2)
        L.append(ground_truth/(10000))
        continue
    
    print("PASSED")
    # np.save("all_ground_truths2.npy", np.array(L))
np.save("all_ground_truths3.npy", np.array(L))
