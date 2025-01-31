import numpy as np
import multiprocessing
import math

def lambert_proj_on_2sphere(point):
    x = point[0]
    y = point[1]
    # print(y)
    new_point = np.zeros((3,))
    new_point[0] = 2*np.cos(2*np.pi*x)*math.sqrt(y - y**2)
    new_point[1] = 2*np.sin(2*np.pi*x)*math.sqrt(y - y**2)
    new_point[2] = 1-2*y
    return new_point

def naive_mapping_to_sphere(point, d):
    x = np.ones((d,))

    for i in range(d-1):
        for j in range(i):
            x[i]*= np.sin(point[j]*2*np.pi)
        x[i]*= np.cos(point[i]*np.pi)
    
    for j in range(d-1):
        x[d-1] *= np.sin(point[j]*np.pi)
        # if(x[d-1] < 0):
        #     x[d-1]*=-1
    # print(point)
    return x

# def random_rotation(d):

def process_theta(theta, X,Y):
    L1 = []
    L2 = []
    temp = 0
    N = X.shape[0]
    # print(N)
    for x in X:
        L1.append(np.dot(x,theta))
    for y in Y:
        L2.append(np.dot(y,theta))

    L1 = sorted(L1)
    L2 = sorted(L2)
    #print("BOYA")
    for i in range(len(L1)):
        temp += (L1[i] - L2[i])**2
    return temp/N


def MY_SW(X,Y, nb_projections, dim):
    SW = 0
    samples = []
    while len(samples) < nb_projections:
        test = np.random.multivariate_normal(np.zeros((dim,)), np.eye(dim), size = 100)
        for theta in test:
            if np.linalg.norm(theta) > 1e-10:
                samples.append(theta/np.linalg.norm(theta))
    print("HOLA")
    #samples = np.array(samples)
    # samples = np.random.normal(0,1,size = (nb_projections, dim))
    # for theta in samples:
    #     norm = np.linalg.norm(theta)
    #     theta = theta/norm
    # for theta in samples:
    #     L1 = []
    #     L2 = []
    #     temp = 0
    #     for x in X:
    #         L1.append(np.dot(x,theta))
    #     for y in Y:
    #         L2.append(np.dot(y,theta))

    #     L1 = sorted(L1)
    #     L2 = sorted(L2)

    #     for i in range(len(L1)):
    #         temp += (L1[i] - L2[i])**2
    #     temp = temp/len(L1)
    #     SW+= temp
        
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_theta, [(theta, X, Y) for theta in samples])
    # SW/= nb_projections
    SW= sum(results) / nb_projections
    return SW





# def MY_SW_2(X,Y, samples):
#     SW = 0
#     # nb_projections = 0
#     nb_projections = samples.shape[0]
#     for theta in samples:
#         L1 = []
#         L2 = []
#         temp = 0
#         for x in X:
#             L1.append(np.dot(x,theta))
#         for y in Y:
#             L2.append(np.dot(y,theta))

#         L1 = sorted(L1)
#         L2 = sorted(L2)

#         for i in range(len(L1)):
#             temp += (L1[i] - L2[i])**2
#         # temp = temp/len(L1)
#         SW+= temp
#         # nb_projections+=1
#     SW/= nb_projections
#     return np.sqrt(SW)

def MY_SW_2(X,Y, samples):
    SW = 0
    # nb_projections = 0
    nb_projections = samples.shape[0]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_theta, [(theta, X, Y) for theta in samples])
    SW= sum(results) / nb_projections
    # print(nb_projections)

    return SW
