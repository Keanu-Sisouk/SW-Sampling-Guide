# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:42:17 2024

@author: f.clement00
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.stats import ortho_group
import ot
import math

from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE

default_steps = 10
default_order = 2
# default_order = -1
default_stepsize = 1e-1
dim_list = [i for i in range(3, 11)]
order_list = [i-2 for i in dim_list]


def normalize(x):
    for i in range(len(x)):
        normalization = np.sqrt(np.sum(x[i] ** 2.0))
        if normalization > 0.0:
            x[i] = x[i] / normalization
        else:
            x[i] = np.random.randn(np.prod(x[i].shape)).reshape(x[i].shape)
            x[i] = x[i] / np.sqrt(np.sum(x[i] ** 2.0))
    return x

def Riesz_noblur_gradient(n, shape, budget=default_steps,order=default_order,step_size=default_stepsize, conv=None):
    t=(n,)+shape
    x = np.random.randn(*t)
    x=normalize(x)
    for steps in range(budget):
        Temp=np.zeros(t)
        for i in range(n):
            for j in range(n):
                 if (j!=i):
                    T=np.add(x[i],-x[j])
                    Temp[i]=np.add(Temp[i],np.multiply(T,1/(np.sqrt(np.sum(T ** 2.0)))**(order+2)))
        x=np.add(x,np.multiply(Temp,step_size))
        x=normalize(x)
    return x


def naive_mapping_to_sphere(point, d):
    x = np.ones((d,))

    for i in range(d-1):
        for j in range(i):
            x[i]*= np.sin(point[j])
        x[i]*= np.cos(point[i])
    
    for j in range(d-1):
        x[d-1] *= np.sin(point[j])
        # if(x[d-1] < 0):
        #     x[d-1]*=-1
    # print(point)
    return x


def lambert_proj_on_2sphere(point):
    x = point[0]
    y = point[1]
    # print(y)
    new_point = np.zeros((3,))
    new_point[0] = 2*np.cos(2*np.pi*x)*math.sqrt(y - y**2)
    new_point[1] = 2*np.sin(2*np.pi*x)*math.sqrt(y - y**2)
    new_point[2] = 1-2*y
    return new_point


# def gen_spherical_fib(N):
#     delta_phi = np.pi*(3 - math.sqrt(5))
#     phi = 0
#     delta_z = 1/N
#     z = 1 - delta_z/2
#     points = np.zeros((N,2))
#     for j in range(N):
#         theta = np.arccos(z)
#         phi_k = phi % (2*np.pi)
#         z -= delta_z
#         phi -= delta_phi
#         points[j][0] = theta
#         points[j][1] = phi_k
#     return points


def fibonacci_sphere(N):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(N):
        y = 1 - (i / float(N - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)


# def gen_random_orthonorm(d):

#     return 0


# N = 500
m = 10
dim = 4

# print(x)




def gen_SW_loss(X,Y,N, d,s):

    sampler = qmc.Halton(d-1, scramble = False)
    sampler2 = qmc.Sobol(d-1, scramble = False)

    points = Riesz_noblur_gradient(N, (1,d), order=s)
    halton_points = sampler.random(N)
    # print(halton_points)
    sobol_points = sampler2.random(N)

    jac_params = 0.5- np.random.rand(d, 2)
    # jac_params = np.array([[0.5, 0.5],
    #                        [-0.3, 0.4]])

    dpp = MultivariateJacobiOPE(N, jac_params)

    dpp_points = dpp.sample()
    dpp_points = (dpp_points + 1)/2

    area_mapped_halton = []
    area_mapped_sobol = []
    area_mapped_dpp = []
    fib_points = None
    SW_5, SW_6, SW_7, SW_9 = 0, 0 ,0, 0
    if(d == 3):
        # print(halton_points)
        # print(new_halton_point)
        for x in halton_points:
            # print(x)
            new_x = lambert_proj_on_2sphere(x)
            area_mapped_halton.append(new_x)
        area_mapped_halton = np.array(area_mapped_halton).T
        # print(new_halton_point.shape)


        for x in sobol_points:
            new_x = lambert_proj_on_2sphere(x)
            area_mapped_sobol.append(new_x)
        area_mapped_sobol = np.array(area_mapped_sobol).T
        # print(new_halton_point.shape)
        # print(area_mapped_sobol.shape)

        fib_points = fibonacci_sphere(N).T
        # print(N)
        # print(fib_points.shape)

        for x in dpp_points:
            new_x = lambert_proj_on_2sphere(x)
            area_mapped_dpp.append(new_x)
        area_mapped_dpp = np.array(area_mapped_dpp).T

    for x in halton_points:
        for i in range(d-2):
            x[i]= x[i]*np.pi
        x[d-2]= x[d-2]*2*np.pi

    for x in sobol_points:
        for i in range(d-2):
            x[i]= x[i]*np.pi
        x[d-2]= x[d-2]*2*np.pi

    for x in dpp_points:
        for i in range(d-2):
            x[i]= x[i]*np.pi
        x[d-2]= x[d-2]*2*np.pi

    new_points = np.zeros((N,d))
    for i in range(N):
        x = points[i][0]
        # if x[d-1] < 0:
        #     x[d-1] = -x[d-1]
        new_points[i] = x
    # new_points = np.array(new_points)

    new_halton_point = []
    # new_halton_point.append(np.array([1,0,0]))
    for x in halton_points:
        new_x = naive_mapping_to_sphere(x, d)
        new_halton_point.append(new_x)
    # print(new_halton_point)
    new_halton_point = np.array(new_halton_point).T
    # print(new_halton_point.shape)

    new_sobol_point = []
    # new_sobol_point.append(np.array([1,0,0]))
    for x in sobol_points:
        new_x = naive_mapping_to_sphere(x, d)
        new_sobol_point.append(new_x)

    new_sobol_point = np.array(new_sobol_point).T

    new_dpp_point = []
    for x in dpp_points:
        new_x = naive_mapping_to_sphere(x, d)
        new_dpp_point.append(new_x)

    new_dpp_point = np.array(new_dpp_point).T

    gauss_sampling = np.random.randn(N, d)
    new_gauss_point = []
    for x in gauss_sampling:
        new_x = x/np.linalg.norm(x)
        new_gauss_point.append(new_x)
    new_gauss_point = np.array(new_gauss_point).T


    orthosamp = []
    while len(orthosamp) < N:
        orthomat = ortho_group.rvs(d)
        # print(orthomat)
        for x in orthomat:
            orthosamp.append(x)
            # print(x)
            # print(np.linalg.norm(x))
            # print(orthosamp)
        # print(len(orthosamp))
        # print(N)
    new_ortho_point = np.array(orthosamp).T
    # print(np.linalg.norm(new_ortho_point.T[0]))
    # print(new_ortho_point.T)
    # print(new_ortho_point.T[0] - new_ortho_point.T[N-1])    
    # print("RAT PASSED!!!!!")

    SW_1 = ot.sliced_wasserstein_distance(X,Y, projections=new_points.T)
    SW_2 = ot.sliced_wasserstein_distance(X,Y, projections=new_halton_point)
    SW_3 = ot.sliced_wasserstein_distance(X,Y, projections=new_sobol_point)
    SW_4 = ot.sliced_wasserstein_distance(X,Y, projections=new_gauss_point)
    SW_8 = ot.sliced_wasserstein_distance(X,Y, projections=new_dpp_point)
    SW_10 = ot.sliced_wasserstein_distance(X,Y, projections=new_ortho_point)
    if (d == 3):
        SW_5 = ot.sliced_wasserstein_distance(X,Y, projections=area_mapped_halton)
        SW_6 = ot.sliced_wasserstein_distance(X,Y, projections=area_mapped_sobol)
        SW_7 = ot.sliced_wasserstein_distance(X,Y, projections=fib_points)
        SW_9 = ot.sliced_wasserstein_distance(X,Y, projections=area_mapped_dpp)
    return SW_1,SW_2,SW_3,SW_4,SW_5,SW_6,SW_7, SW_8, SW_9, SW_10



N_list = [i for i in range(10,250)]





# dim_list = [3]



all_SW1 = []
all_SW2 = []
all_SW3 = []
all_SW4 = []
all_SW5 = []
all_SW6 = []
all_SW7 = []
all_SW8 = []
all_SW9 = []
all_SW10 = []

for i in range(len(dim_list)):
# for d in dim_list:

    print("RUN YOU RAT RUN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    d = dim_list[i]
    X = 100*np.random.randn(1000, d)
    Y = 100*np.random.randn(1000, d)
    s = default_order
    if i == 0:
        s = order_list[i]
    L_SW1 = []
    L_SW2 = []
    L_SW3 = []
    L_SW4 = []
    L_SW8 = []
    L_SW10 = []
    if (d== 3):
        L_SW5 = []
        L_SW6 = []
        L_SW7 = []
        L_SW9 = []
        for i in range(len(N_list)):
            SW_1,SW_2,SW_3,SW_4,SW_5,SW_6,SW_7, SW_8, SW_9, SW_10 = gen_SW_loss(X,Y,N_list[i], d,s)
            L_SW1.append(SW_1)
            L_SW2.append(SW_2)
            L_SW3.append(SW_3)
            L_SW4.append(SW_4)
            L_SW5.append(SW_5)
            L_SW6.append(SW_6)
            L_SW7.append(SW_7)
            L_SW8.append(SW_8)
            L_SW9.append(SW_9)
            L_SW10.append(SW_10)
        all_SW5.append(L_SW5)
        all_SW6.append(L_SW6)
        all_SW7.append(L_SW7)
        all_SW9.append(L_SW9)
        # all_SW10.append(L_SW10)
    else:          
        for i in range(len(N_list)):
            SW_1,SW_2,SW_3,SW_4,_,_,_,SW_8,_,SW10 = gen_SW_loss(X,Y,N_list[i], d,s)
            L_SW1.append(SW_1)
            L_SW2.append(SW_2)
            L_SW3.append(SW_3)
            L_SW4.append(SW_4)
            L_SW8.append(SW_8)
            L_SW10.append(SW_10)
    all_SW1.append(L_SW1)
    all_SW2.append(L_SW2)
    all_SW3.append(L_SW3)
    all_SW4.append(L_SW4)
    all_SW8.append(L_SW8)
    all_SW10.append(L_SW10)



for i in range(len(dim_list)):
    plt.plot(N_list, all_SW1[i], "red", marker = "o", label = "Riez energy")
    plt.plot(N_list, all_SW2[i], "blue", marker = "+",label = "Halton points mapped to sphere")
    plt.plot(N_list, all_SW3[i], "green",label = "Sobol points mapped to sphere")
    plt.plot(N_list, all_SW4[i], "brown", marker = "^",label = "Random uniform on the sphere")
    plt.plot(N_list, all_SW8[i], "pink", marker = "v",label = "DPP mapped on the sphere")
    plt.plot(N_list, all_SW10[i], "yellow", marker = ">",label = "Orthonormal sampling on the sphere")
    if dim_list[i] == 3:
        plt.plot(N_list, all_SW5[i], "black", marker = ".",label = "Halton points area mapped to sphere")
        plt.plot(N_list, all_SW6[i], "purple", marker = ",",label = "Sobol points area mapped to sphere")
        plt.plot(N_list, all_SW7[i], "orange", marker = "<",label = "Fibonacci points on the sphere")
        plt.plot(N_list, all_SW9[i], "turquoise", marker = "*",label = "DPP area mapped to sphere")
    plt.title("For measures in " + str(dim_list[i])+ " dimension")
    plt.xlabel("Number of projections")
    plt.ylabel("SW_2")
    plt.legend()
    plt.show()

# print(new_gauss_point.shape)

# print(new_points)
# print(points.shape)
# print(points[0])
# for x in points:
# for x in points:
#     p = x[0]
#     plt.scatter(p[0], p[1],p[2])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# ax.scatter(new_points[:,0], new_points[:,1], new_points[:,2])
# plt.show()
# plt.close()

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection = "3d")
# ax2.scatter(new_halton_point[:,0], new_halton_point[:,1], new_halton_point[:,2])
# plt.show()
# plt.close()

# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111, projection = "3d")
# ax3.scatter(new_sobol_point[:,0], new_sobol_point[:,1], new_sobol_point[:,2])
# plt.show()
# plt.close()

# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111, projection = "3d")
# ax4.scatter(new_gauss_point[:,0], new_gauss_point[:,1], new_gauss_point[:,2])
# plt.show()
# plt.close()










