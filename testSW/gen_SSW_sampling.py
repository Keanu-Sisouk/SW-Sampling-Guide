import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn.functional as F

import ot
import time

dim_list = [2,3,5,10,20,50]


N_list_temp = [n for n in range(100,10101,1000)]
N_list = [N_list_temp[0]] + [300,500,700,900] + N_list_temp[1:]
# N_list = [8100,9100,10100]
# trash = np.load("all_ground_truths2.npy")
# print(trash.shape)
ground_truths = np.load("all_ground_truths3.npy")



def gen_SSW_sampling(N, d):
    x0 = torch.randn(N, d)
    x0 = F.normalize(x0, dim=-1)

    x = x0.clone()
    x.requires_grad_(True)

    n_iter = 250
    lr = 150

    # losses = []
    # xvisu = torch.zeros(n_itr)

    number_of_proj = 500


    for i in range(n_iter):
        sw = ot.sliced_wasserstein_sphere_unif(x, n_projections = number_of_proj )
        grad_x = torch.autograd.grad(sw,x)[0]

        x = x - lr * grad_x / np.sqrt(i / 10 + 1)
        x = F.normalize(x, p = 2, dim = 1)

        # losses.append(sw.item())
    return x

for j in range(len(dim_list)):

    if j == 0:
        continue

    d = dim_list[j]
    all_lists = []
    # agent = SamplerAgent(d)
    # agent.SetNewRieszParam(d - 2, 20, 5e-2)

    X = np.load("X_"+str(d)+"D.npy")
    Y = np.load("Y_"+str(d)+"D.npy")

    # print(type(X))
    # print(type(Y))

    ground_truth = ground_truths[j]

    timers_list = []
    errors_list = []
    for N in N_list:
        t0 = time.time()
        points = gen_SSW_sampling(N,d).detach().clone().numpy()
        SW = pow(ot.sliced_wasserstein_distance(X,Y,projections = points.T), 2)
        t1 = time.time()
        total = t1 - t0
        # timers_list.append(total)
        # SW = MY_SW_2(X,Y,nppoints)
        # errors_list.append(abs(SW - ground_truth))
        np.save("First_SSW_"+str(N)+"points_in_"+str(d)+"D.npy", points)
    
    # print(d)
    #     # np.save("first_"+str(N)+"_riesz_points_in_"+str(d)+"D.npy", new_points)
    # # print("PASSED DIMENSION" + str(d))
    # np.save("timers_SSW_in_"+str(d)+"D.npy", np.array(timers_list))
    # np.save("errors_SSW_in_"+str(d)+"D.npy", np.array(errors_list))
