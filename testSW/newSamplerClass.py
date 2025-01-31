import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.stats import ortho_group
import ot
import math

from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE


class SamplerAgent:
    dim = 3
    default_steps = 10
    default_order_riesz = 0
    default_stepsize = 1e-1

    # LDSScrambler = False

    def __init__(self, dim):
        self.dim = dim
        self.sobol_sampler = qmc.Sobol(d = self.dim - 1, scramble = False)
        self.halton_sampler = qmc.Halton(d = self.dim - 1, scramble = False)
        self.sobol_sampler_rand = qmc.Sobol(d = self.dim - 1, scramble = True)
        self.halton_sampler_rand = qmc.Halton(d = self.dim - 1, scramble = True)
        self.jac_params = 0.5- np.random.rand(dim, 2)

        
    def SetNewRieszParam(self, order, steps, step_size):
        self.default_steps = steps
        self.default_order_riesz = order
        self.default_stepsize = step_size

    
    def normalize(self,x):
        for i in range(len(x)):
            normalization = np.sqrt(np.sum(x[i] ** 2.0))
            if normalization > 0.0:
                x[i] = x[i] / normalization
            else:
                x[i] = np.random.randn(np.prod(x[i].shape)).reshape(x[i].shape)
                x[i] = x[i] / np.sqrt(np.sum(x[i] ** 2.0))
        return x


    def Riesz_noblur_gradient(self,n, budget=default_steps,order=default_order_riesz,step_size=default_stepsize, conv=None):
        shape = (1,self.dim)
        
        t=(n,)+shape
        x = np.random.randn(*t)
        x=self.normalize(x)
        for steps in range(budget):
            Temp=np.zeros(t)
            for i in range(n):
                for j in range(n):
                    if (j!=i):
                        T=np.add(x[i],-x[j])
                        Temp[i]=np.add(Temp[i],np.multiply(T,1/(np.sqrt(np.sum(T ** 2.0)))**(order+2)))
            x=np.add(x,np.multiply(Temp,step_size))
            x=self.normalize(x)
        return x

    def Riesz_noblur_gradient(self,n, budget=default_steps,order=default_order_riesz,step_size=default_stepsize, conv=None):
        shape = (1,self.dim)

        t=(n,)+shape
        x = np.random.randn(*t)
        x=self.normalize(x)
        for steps in range(budget):
            Temp=np.zeros(t)
            for i in range(n):
                for j in range(n):
                    if (j!=i):
                        T=np.add(x[i],-x[j])
                        Temp[i]=np.add(Temp[i],np.multiply(T,1/(np.sqrt(np.sum(T ** 2.0)))**(order+2)))
            x=np.add(x,Temp)
            x=normalize(x)
        return x


    def gen_sobol(self,N, sobol_list):
        lenlist = len(sobol_list)
        sobol_points = self.sobol_sampler.random_base2(m=10)[lenlist:N]
        # for x in sobol_points:
        #     sobol_list.append(x)
        return sobol_points
    
    def gen_halton(self,N, halton_list):
        lenlist = len(halton_list)
        halton_points = self.halton_sampler.random(N-lenlist)
        # for x in halton_points:
        #     halton_list.append(x)
        return halton_points

    def gen_sobol_rand(self,N, rand_sobol_list):
        lenlist = len(rand_sobol_list)

        rand_sobol_points = self.sobol_sampler_rand.random_base2(m=10)[lenlist:N]
        # for x in rand_sobol_points:
        #     rand_sobol_list.append(x)
        return rand_sobol_points

    def gen_halton_rand(self,N, rand_halton_list):
        lenlist = len(rand_halton_list)
        rand_halton_points = self.halton_sampler_rand.random(N-lenlist)
        # for x in rand_halton_points:
        #     rand_halton_list.append(x)
        return rand_halton_points

    def gen_dppy(self,N):
        # jac_params = 0.5- np.random.rand(self.dim, 2)
        dpp = MultivariateJacobiOPE(N, self.jac_params)

        dpp_points = dpp.sample()
        dpp_points = (dpp_points + 1)/2
        return dpp_points
        
    def fibonacci_sphere(self,N):
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

    def uniform_sampling(self,N,list_gauss_sampling):
        # gauss_sampling = np.random.randn(N, self.dim)
        # new_gauss_point = []
        # for x in gauss_sampling:
        #     new_x = x/np.linalg.norm(x)
        #     new_gauss_point.append(new_x)
        # new_gauss_point = np.array(new_gauss_point)
        # return new_gauss_point

        cov = np.eye(self.dim)
        while len(list_gauss_sampling) <= N:
            gauss_sampling = np.random.multivariate_normal(np.zeros((self.dim,)), cov, size = 100)
            # print(gauss_sampling.shape)
            for x in gauss_sampling:
                if np.linalg.norm(x) > 1e-10:
                    new_x = x/np.linalg.norm(x)
                    # print(np.linalg.norm(new_x))
                    # print(new_x.shape)
                    # new_gauss_point.append(new_x)
                    list_gauss_sampling.append(new_x)
        # new_gauss_point = np.array(new_gauss_point)
        return list_gauss_sampling

    def ortho_sampling(self,N, orthosamp):
        while len(orthosamp) < N:
            orthomat = ortho_group.rvs(self.dim)
            # print(orthomat)
            for x in orthomat.T:
                orthosamp.append(x)
        # new_ortho_point = np.array(orthosamp)
        # return new_ortho_point
        return orthosamp