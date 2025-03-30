import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.stats import ortho_group
import ot
import math

from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE

from samplerclass import *
from utils import *
from sliced_wasserstein import SW_QMC, SW_RQMC

# import time

dim_list = [2,3,5,10,20,50]


N_list_temp = [n for n in range(100,10101,1000)]
N_list = [N_list_temp[0]] + [300,500,700,900] + N_list_temp[1:]
# ground_truths = np.load("all_ground_truths.npy")
ground_truths = np.load("all_ground_truths3.npy")


for j in range(len(dim_list)):
# for d in dim_list:
    # mean_error_list = []
    if j == 0:
        continue
    d = dim_list[j]
    all_lists = []
    agent = SamplerAgent(d)

    X = np.load("X_"+str(d)+"D.npy")
    Y = np.load("Y_"+str(d)+"D.npy")

    # print(type(X))
    # print(type(Y))

    # ground_truth = MY_SW(X,Y, 1000, d)
    ground_truth = ground_truths[j]

    timers_list = []
    errors_list = []

    new_points = []
    for i in range(len(N_list)):
        # t0 = time.time()
        # points = agent.gen_halton(N_list[i], new_points)
        # t1 = time.time()
        # total = t1 - t0

        # for x in points:
        #     new_x = naive_mapping_to_sphere(x, d)
        #     new_points.append(new_x)

        # if len(timers_list) > 0:
        #     timers_list.append(total + timers_list[i-1])
        # else:
        #     timers_list.append(total)

        # new_points = np.zeros((N,d))
        # for i in range(N):
        #     x = points[i][0]
        #     # if x[d-1] < 0:
        #     #     x[d-1] = -x[d-1]
        #     new_points[i] = x
        # new_points = np.array(new_points)

        # SW = pow(ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T),2)
        SW, total = SW_QMC(X,Y, L = N_list[i], seq="sobol")
        # SW = MY_SW_2(X,Y,np.array(new_points))
        errors_list.append(abs(SW - ground_truth))
        timers_list.append(total)
    
    np.save("timers_sobol_leluc_in_"+str(d)+"D.npy", np.array(timers_list))
    np.save("errors_sobol_leluc_in_"+str(d)+"D.npy", np.array(errors_list))

# np.save("N_list.npy", np.array(N_list))
