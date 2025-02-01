import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.stats import ortho_group
import ot
import math
import random

from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from spherical_harmonics import SphericalHarmonics


from samplerclass import *
from utils import *
from sliced_wasserstein import SHCV
import time

dim_list = [2,3,5,10,20,50]

N_list_temp = [n for n in range(100,10101,1000)]
N_list = [N_list_temp[0]] + [300,500,700,900] + N_list_temp[1:] + [n for n in range(20000, 50001, 10000)]
# ground_truths = np.load("all_ground_truths.npy")
ground_truths = np.load("all_ground_truths3.npy")


for j in range(len(dim_list)):
# for d in dim_list:
    # mean_error_list = []
    if j == 0 or j == 5:
        continue


    d = dim_list[j]
    all_lists = []
    MAX_DEG = 0
    # agent = SamplerAgent(d)
    if d==2:
        MAX_DEG = 17
    if d==3:
        MAX_DEG = 17
    if d==5:
        MAX_DEG = 7
    if d==10:
        MAX_DEG = 5
    if d==20:
        MAX_DEG = 4
    if d==50:
        MAX_DEG = 4

    Phi = SphericalHarmonics(dimension=d,degrees=MAX_DEG)

    X = np.load("X_"+str(d)+"D.npy")
    Y = np.load("Y_"+str(d)+"D.npy")

    # print(type(X))
    # print(type(Y))

    # ground_truth = MY_SW(X,Y, 1000, d)
    ground_truth = ground_truths[j]

    for p in range(100):
        # print(ground_truth)
        timers_list = []
        errors_list = []
        new_points = []
        # SW_list = []
        prev_SW = 0
        prev_len = 0
        for i in range(len(N_list)):
            # new_points = agent.ortho_sampling(N_list[i], new_points)
            
            
            # if len(timers_list) > 0:
            #     timers_list.append(total + timers_list[i-1])
            # else:
            #     timers_list.append(total)
            # print(len(new_points))
            # SW = ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T)
            # diff = len(new_points) - prev_len
            # SW = ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T)
            # SW = MY_SW_2(X,Y,np.array(new_points))
            # SW = (prev_len*prev_SW + diff*MY_SW_2(X,Y,np.array(new_points[prev_len:])))/len(new_points)
            # SW = (prev_len*prev_SW + diff*pow(ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points[prev_len:]).T),2))/len(new_points)
            # print(SW)
            seed = random.randint(0,100000000)
            SW, total = SHCV(X, Y, seed,L=N_list[i], p=2, Phi=Phi)
            errors_list.append(abs(SW - ground_truth))
            # SW_list.append(SW)
            prev_SW = SW
            prev_len = len(new_points)

            timers_list.append(total)
            # SW_list.append(SW)
        all_lists.append(errors_list)
        np.save("timers_sphereharmonics_SWIncluded_PointsStored_in_"+str(d)+"D.npy", np.array(timers_list))

    mean_error_list = [0 for i in range(len(all_lists[0]))]
    for L in all_lists:
        for i in range(len(L)):
            mean_error_list[i]+= L[i]
    
    for i in range(len(mean_error_list)):
        mean_error_list[i] = mean_error_list[i]/100
    

    temp = np.load("errors_sphereharmonics_SWIncluded_in_"+str(d)+"D.npy")    
    temp2 = [x for x in temp] + mean_error_list[15:]

    np.save("errors_sphereharmonics_SWIncluded_PointsStored_in_"+str(d)+"D.npy", np.array(temp2))

# np.save("N_list.npy", np.array(N_list))
