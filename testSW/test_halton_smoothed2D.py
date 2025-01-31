import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.stats import ortho_group
import ot
import math
from scipy.stats import special_ortho_group


from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE

from samplerclass import *
from utils import *
import time

dim_list = [2,3,5,10,20,50]


N_list_temp = [n for n in range(100,10101,1000)]
N_list = [N_list_temp[0]] + [300,500,700,900] + N_list_temp[1:]

d= 2
all_lists = []

X = np.load("X_"+str(d)+"D.npy")
Y = np.load("Y_"+str(d)+"D.npy")

# print(type(X))
# print(type(Y))

# all_ground_truths = np.load("all_ground_truths.npy")
all_ground_truths = np.load("all_ground_truths3.npy")


ground_truth = all_ground_truths[0]
for p in range(100):
    agent = SamplerAgent(d)
    orthomat = special_ortho_group.rvs(d)
    timers_list = []
    errors_list = []
    new_points = []
    prev_SW = 0
    prev_len = 0

    for i in range(X.shape[0]):
        X[i] = np.dot(orthomat, X[i])
        Y[i] = np.dot(orthomat, Y[i])

    for i in range(len(N_list)):

        t0 = time.time()
        points = agent.gen_halton(N_list[i], new_points)



        for x in points:
            # print(x)
            # new_x = lambert_proj_on_2sphere(x)
            # print(x)
            new_x = [np.cos(np.pi*x[0]), np.sin(np.pi*x[0])]
            # print(np.linalg.norm(np.array(new_x)))
            new_points.append(new_x)




        # new_points = np.zeros((N,d))
        # for i in range(N):
        #     x = points[i][0]
        #     # if x[d-1] < 0:
        #     #     x[d-1] = -x[d-1]
        #     new_points[i] = x
        # new_points = np.array(new_points)

        # SW = ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T)
        diff = len(new_points) - prev_len
        # SW = ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T)
        # SW = MY_SW_2(X,Y,np.array(new_points))
        # SW = (prev_len*prev_SW + diff*MY_SW_2(X,Y,np.array(new_points[prev_len:])))/len(new_points)
        SW = (prev_len*prev_SW + diff*pow(ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points[prev_len:]).T),2))/len(new_points)
        t1 = time.time()
        total = t1 - t0

        if len(timers_list) > 0:
            timers_list.append(total + timers_list[i-1])
        else:
            timers_list.append(total)
        errors_list.append(abs(SW - ground_truth))
        # SW_list.append(SW)
        prev_SW = SW
        prev_len = len(new_points)
    all_lists.append(errors_list)
    np.save("timers_haltonarea_smoothed_SWIncluded_in_"+str(d)+"D.npy", np.array(timers_list))

mean_error_list = [0 for i in range(len(all_lists[0]))]
for L in all_lists:
    for i in range(len(L)):
        mean_error_list[i]+= L[i]

for i in range(len(mean_error_list)):
    mean_error_list[i] = mean_error_list[i]/100

np.save("errors_haltonarea_smoothed_SWIncluded_in_"+str(d)+"D.npy", np.array(mean_error_list))

np.save("N_list.npy", np.array(N_list))
