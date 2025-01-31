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

# dim_list = [i for i in range(3, 100, 10)]
dim_list = [2,3,5,10,20,50]


N_list_temp = [n for n in range(100,10101,1000)]
N_list = [N_list_temp[0]] + [300,500,700,900] + N_list_temp[1:]

# ground_truths = np.load("all_ground_truths.npy")
ground_truths = np.load("all_ground_truths3.npy")

# print(ground_truths)
for j in range(len(ground_truths)):
# for d in dim_list:
    # mean_error_list = []
    d = dim_list[j]
    all_lists = []
    agent = SamplerAgent(d)
    X = np.load("X_"+str(d)+"D.npy")
    Y = np.load("Y_"+str(d)+"D.npy")

    # print(type(X))
    # print(type(Y))

    # ground_truth = MY_SW(X,Y, 1000, d)
    ground_truth = ground_truths[j]
    # print("bouh")
    for p in range(100):
        # print(ground_truth)
        timers_list = []
        errors_list = []
        new_points = []
        # SW_list = []
        prev_SW = 0
        prev_len = 0
        if (p % 5 == 0):
            # print("BOUYA1")

        for i in range(len(N_list)):

            t0 = time.time()
            new_points = agent.uniform_sampling(N_list[i], new_points)
            # t1 = time.time()
            # total = t1 - t0
            # new_points = np.load("first_"+str(N_list[i])+"_unif_points_in_"+str(d)+"D_version_"+str(p)+".npy")

            # if len(timers_list) > 0:
            #     timers_list.append(total + timers_list[i-1])
            # else:
            #     timers_list.append(total)
            # print(len(new_points))
            diff = len(new_points) - prev_len
            # SW = ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T)
            # SW = (prev_len*prev_SW + diff*MY_SW_2(X,Y,np.array(new_points[prev_len:])))/len(new_points)
            SW = (prev_len*prev_SW + diff*pow(ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points[prev_len:]).T),2))/len(new_points)
            t1 = time.time()
            total = t1 - t0
            # print(SW)

            if len(timers_list) > 0:
                timers_list.append(total + timers_list[i-1])
            else:
                timers_list.append(total)
            errors_list.append(abs(SW - ground_truth))
            # SW_list.append(SW)
            prev_SW = SW
            prev_len = len(new_points)

        all_lists.append(errors_list)
        np.save("timers_unif_SWIncluded_PointsStored_in_"+str(d)+"D.npy", np.array(timers_list))
        # print(p)
    mean_error_list = [0 for i in range(len(all_lists[0]))]
    for L in all_lists:
        for i in range(len(L)):
            mean_error_list[i]+= L[i]
    
    for i in range(len(mean_error_list)):
        mean_error_list[i] = mean_error_list[i]/100
    
    # plt.plot(mean_error_list)
    # plt.show()



    # plt.plot(SW_list)
    # plt.show()
    # plt.plot(errors_list)
    # plt.show()

    # print("RUNNING")
    # print(timers_list)
    temp = np.load("errors_unif_SWIncluded_in_"+str(d)+"D.npy")    
    temp2 = [x for x in temp] + mean_error_list[15:]

    np.save("errors_unif_SWIncluded_PointsStored_in_"+str(d)+"D.npy", np.array(temp2))

np.save("N_list.npy", np.array(N_list))


# # for i in range(len(dim_list)):
# for j in range(len(ground_truths)):
# # for d in dim_list:
#     # mean_error_list = []
#     d = dim_list[j]
#     all_lists = []
#     agent = SamplerAgent(d)

#     X = np.load("X_"+str(d)+"D.npy")
#     Y = np.load("Y_"+str(d)+"D.npy")

#     # print(type(X))
#     # print(type(Y))

#     # ground_truth = MY_SW(X,Y, 1000, d)
#     ground_truth = ground_truths[j]
#     # print("bouh")

#     for p in range(100):
#         # print(ground_truth)
#         timers_list = []
#         errors_list = []
#         new_points = []
#         # SW_list = []
#         prev_SW = 0
#         prev_len = 0
#         if (p % 5 == 0):
#             print("BOUYA2")
#         for i in range(len(N_list)):
            
#             t0 = time.time()
#             new_points = agent.ortho_sampling(N_list[i], new_points)
#             t1 = time.time()
#             total = t1 - t0
#             if len(timers_list) > 0:
#                 timers_list.append(total + timers_list[i-1])
#             else:
#                 timers_list.append(total)
#             # print(len(new_points))
#             # SW = ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T)
#             diff = len(new_points) - prev_len
#             # SW = ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T)
#             # SW = MY_SW_2(X,Y,np.array(new_points))
#             SW = (prev_len*prev_SW + diff*pow(ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points[prev_len:]).T),2))/len(new_points)
#             # print(SW)
#             errors_list.append(abs(SW - ground_truth))
#             # SW_list.append(SW)
#             prev_SW = SW
#             prev_len = len(new_points)
#             # SW_list.append(SW)
#         all_lists.append(errors_list)
#         np.save("timers_ortho_in_"+str(d)+"D.npy", np.array(timers_list))

#     mean_error_list = [0 for i in range(len(all_lists[0]))]
#     for L in all_lists:
#         for i in range(len(L)):
#             mean_error_list[i]+= L[i]
    
#     for i in range(len(mean_error_list)):
#         mean_error_list[i] = mean_error_list[i]/100
    
#     np.save("errors_ortho_in_"+str(d)+"D.npy", np.array(mean_error_list))

# np.save("N_list.npy", np.array(N_list))


# for j in range(len(ground_truths)):
# # for d in dim_list:
#     # mean_error_list = []
#     d = dim_list[j]
#     all_lists = []
#     agent = SamplerAgent(d)

#     X = np.load("X_"+str(d)+"D.npy")
#     Y = np.load("Y_"+str(d)+"D.npy")

#     # print(type(X))
#     # print(type(Y))

#     # ground_truth = MY_SW(X,Y, 1000, d)
#     ground_truth = ground_truths[j]
#     for p in range(100):

#         timers_list = []
#         errors_list = []
#         new_points = []
#         prev_SW = 0
#         prev_len = 0
#         print("BOUYA3")
#         for i in range(len(N_list)):
            

#             t0 = time.time()
#             points = agent.gen_halton_rand(N_list[i], new_points)
#             for x in points:
#                 new_x = naive_mapping_to_sphere(x, d)
#                 new_points.append(new_x)
#             t1 = time.time()
#             total = t1 - t0


#             if len(timers_list) > 0:
#                 timers_list.append(total + timers_list[i-1])
#             else:
#                 timers_list.append(total)

#             # new_points = np.zeros((N,d))
#             # for i in range(N):
#             #     x = points[i][0]
#             #     # if x[d-1] < 0:
#             #     #     x[d-1] = -x[d-1]
#             #     new_points[i] = x
#             # new_points = np.array(new_points)

#             # SW = ot.sliced_wasserstein_distance(X,Y, projections=np.array(new_points).T)
#             # SW = MY_SW_2(X,Y,np.array(new_points))
#             diff = len(new_points) - prev_len

#             SW = (prev_len*prev_SW + diff*MY_SW_2(X,Y,np.array(new_points[prev_len:])))/len(new_points)
#             errors_list.append(abs(SW - ground_truth))
#             prev_SW = SW
#             prev_len = len(new_points)

#         all_lists.append(errors_list)

#         np.save("timers_sobolrand_in_"+str(d)+"D.npy", np.array(timers_list))

#     mean_error_list = [0 for i in range(len(all_lists[0]))]
#     for L in all_lists:
#         for i in range(len(L)):
#             mean_error_list[i]+= L[i]
    
#     for i in range(len(mean_error_list)):
#         mean_error_list[i] = mean_error_list[i]/100
    
#     # np.save("timers_sobolrand_in_"+str(d)+"D.npy", np.array(timers_list))
#     np.save("errors_sobolrand_in_"+str(d)+"D.npy", np.array(mean_error_list))
#     plt.plot(mean_error_list)
#     plt.show()

# np.save("N_list.npy", np.array(N_list))




