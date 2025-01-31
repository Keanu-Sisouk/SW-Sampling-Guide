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

from samplerclass import *
from utils import *

NbPoints = 1000

# dim_list = [i for i in range(3, 100, 10)]

# dim_list = [2,3,5,10,20]
dim_list = [2,3,5,10,20,50,100]
# N_list = np.linspace(100,10000,25)

for d in dim_list:
    mu1 = np.random.multivariate_normal(np.ones((d,)), np.eye(d), 1)
    mu2 = np.random.multivariate_normal(np.ones((d,)), np.eye(d), 1)

    temp1 = np.zeros((d,d))
    temp2 = np.zeros((d,d))

    for i in range(d):
        for j in range(d):
            temp1[i][j] = np.random.normal()
            temp2[i][j] = np.random.normal()

    cov1 = np.dot(temp1, temp1.T)
    cov2 = np.dot(temp2, temp2.T)

    X = np.random.multivariate_normal(mu1[0], cov1, NbPoints)
    Y = np.random.multivariate_normal(mu2[0], cov2, NbPoints)

    np.save("X_"+str(d)+"D.npy", X)
    np.save("Y_"+str(d)+"D.npy", Y)


    # agent = samplerAgent(d)

    # timers_unif = []
    # timers_ortho = []
    # timers_sobol = []
    # timers_halton = []
    # timers_riesz = []
    # timers_rand_sobol = []
    # timers_rand_halton = []
    # timers_ = []
    # timers_unif = []
    # timers_unif = []





