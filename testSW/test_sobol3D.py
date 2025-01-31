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
import time

dim_list = [i for i in range(3, 100, 10)]


N_list_temp = [n for n in range(100,10101,1000)]
N_list = [N_list_temp[0]] + [300,500,700,900] + N_list_temp[1:]

d = 3
agent = SamplerAgent(d)

X = np.load("X_"+str(d)+"D.npy")
Y = np.load("Y_"+str(d)+"D.npy")

# print(type(X))
# print(type(Y))
# all_ground_truths = np.load("all_ground_truths.npy")
all_ground_truths = np.load("all_ground_truths2.npy")


ground_truth = all_ground_truths[1]

timers_list = []
errors_list = []
new_points = []

# for N in N_list:
for i in range(len(N_list)):

    t0 = time.time()
    points = agent.gen_sobol(N_list[i], new_points)
    t1 = time.time()
    total = t1 - t0

    if len(timers_list) > 0:
        timers_list.append(total + timers_list[i-1])
    else:
        timers_list.append(total)

    # new_points = np.zeros((N,d))
    for x in points:
        new_x = lambert_proj_on_2sphere(x)
        new_points.append(new_x)

    SW = pow(ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T),2)
    # SW = MY_SW_2(X,Y,np.array(new_points))
    errors_list.append(abs(SW - ground_truth))
    
    np.save("new_timers_sobolarea_in_"+str(d)+"D.npy", np.array(timers_list))
    np.save("new_errors_sobolarea_in_"+str(d)+"D.npy", np.array(errors_list))

np.save("N_list.npy", np.array(N_list))
