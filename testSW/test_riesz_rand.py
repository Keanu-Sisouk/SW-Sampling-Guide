import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.stats import ortho_group
from scipy.stats import special_ortho_group

import ot
import math

from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE

from samplerclass import *
from utils import *
import time

dim_list = [2,3,5,10,20,50]


N_list_temp = [n for n in range(100,10101,1000)]
N_list = [N_list_temp[0]] + [300,500,700,900] + N_list_temp[1:]
# ground_truths = np.load("all_ground_truths2.npy")
ground_truths = np.load("all_ground_truths3.npy")


for j in range(len(dim_list)):
    # if j > 0:
    #     break
    d = dim_list[j]
    all_lists = []
    agent = SamplerAgent(d)
    # agent.SetNewRieszParam(d - 2, 20, 5e-2)

    X = np.load("X_"+str(d)+"D.npy")
    Y = np.load("Y_"+str(d)+"D.npy")

    # print(type(X))
    # print(type(Y))

    ground_truth = ground_truths[j]
    for p in range(100):
        orthomat = special_ortho_group.rvs(d)
        # np.save("orthomat_"+str(p)+"_in_"+str(d)+"D.npy", orthomat)
        timers_list = []
        errors_list = []
        if d == 2:
             for N in N_list:
                 t0 = time.time()
                 points = np.zeros((N,2))
                 for i in range(N):
                     points[i][0] = np.cos(np.pi*i/N)
                     points[i][1] = np.sin(np.pi*i/N)
                     points[i] = np.dot(orthomat, points[i])
                 t1 = time.time()
        
                 total = t1 - t0

                 timers_list.append(total)

                 SW = pow(ot.sliced_wasserstein_distance(X,Y, projections=points.T),2)
                 # SW = MY_SW_2(X,Y,points)
                 errors_list.append(abs(SW - ground_truth))
        else:
            for N in N_list:
                t0 = time.time()
                # points = agent.Riesz_noblur_gradient(n=N, 10, 0.01 , 1e1)
                # total = t1 - t0
                points = np.load("new_first_"+str(N)+"_riesz_points_in_"+str(d)+"D.npy")
                t1 = time.time()

                timers_list.append(total)

                new_points = np.zeros((N,d))
                for i in range(N):
                    x = points[i]
                    # if x[d-1] < 0:
                    #     x[d-1] = -x[d-1]
                    new_points[i] = np.dot(orthomat,x)
                new_points = np.array(new_points)

                SW = pow(ot.sliced_wasserstein_distance(X,Y, projections=new_points.T),2)

                errors_list.append(abs(SW - ground_truth))
        all_lists.append(errors_list)
        # np.save("timers_haltonrand_in_"+str(d)+"D.npy", np.array(timers_list))

    mean_errors_list = [0 for i in range(len(all_lists[0]))]
    for L in all_lists:
        for i in range(len(L)):
            mean_errors_list[i]+= L[i]

    for i in range(len(mean_errors_list)):
        mean_errors_list[i] = mean_errors_list[i]/100
    # print("PASSED DIMENSION" + str(d))
    # np.save("timers_riesz_secver_in_"+str(d)+"D.npy", np.array(timers_list))

    np.save("errors_riesz_rand_in_"+str(d)+"D.npy", np.array(mean_errors_list))

# np.save("N_list.npy", np.array(N_list))
