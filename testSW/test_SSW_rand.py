import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn.functional as F

from scipy.stats import ortho_group
from scipy.stats import special_ortho_group


import ot
import time
dim_list = [2,3,5,10,20,50]


N_list_temp = [n for n in range(100,10101,1000)]
N_list = [N_list_temp[0]] + [300,500,700,900] + N_list_temp[1:]
# N_list = [8100,9100,10100]
# ground_truths = np.load("all_ground_truths2.npy")
ground_truths = np.load("all_ground_truths3.npy")


for j in range(len(dim_list)):
    if j == 0:
        continue
    d = dim_list[j]
    all_lists = []

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

        for N in N_list:
            # t0 = time.time()
            # points = agent.Riesz_noblur_gradient(n=N)
            # total = t1 - t0
            points = np.load("First_SSW_"+str(N)+"points_in_"+str(d)+"D.npy")
            # t1 = time.time()

            # timers_list.append(total)

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

    np.save("errors_SSWrand_in_"+str(d)+"D.npy", np.array(mean_errors_list))

# np.save("N_list.npy", np.array(N_list))
