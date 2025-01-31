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
d= 3
agent = SamplerAgent(d)

X = np.load("X_"+str(d)+"D.npy")
Y = np.load("Y_"+str(d)+"D.npy")

# print(type(X))
# print(type(Y))

# all_ground_truths = np.load("all_ground_truths.npy")
all_ground_truths = np.load("all_ground_truths3.npy")


ground_truth = all_ground_truths[1]

timers_list = []
errors_list = []
for N in N_list:
    t0 = time.time()
    points = agent.fibonacci_sphere(N)
    # total = t1 - t0

    # timers_list.append(total)

    # new_points = np.zeros((N,d))
    # for i in range(N):
    #     x = points[i][0]
    #     # if x[d-1] < 0:
    #     #     x[d-1] = -x[d-1]
    #     new_points[i] = x


    SW = pow(ot.sliced_wasserstein_distance(X,Y, projections=points.T),2)
    # SW = MY_SW_2(X,Y,points)
    t1 = time.time()

    total = t1 - t0

    timers_list.append(total)

    errors_list.append(abs(SW - ground_truth))

np.save("timers_fib_SWIncluded_in_"+str(d)+"D.npy", np.array(timers_list))
np.save("errors_fib_SWIncluded_in_"+str(d)+"D.npy", np.array(errors_list))

np.save("N_list.npy", np.array(N_list))
