# -*- coding: utf-8 -*-
"""
Implementation of Mixed Membership Stochatic Blockmodel using Variationl Inference from:
http://jmlr.csail.mit.edu/papers/volume9/airoldi08a/airoldi08a.pdf

Created on Mon Jul 17 15:41:24 2017

@author: Saeid Parvandeh
"""
from __future__ import division
import numpy as np
import pymc
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
import itertools

# this function creates a stochastic block model using an observed graph
# from section 2-page 1984
def create_model(data_matrix, num_people, num_groups, alpha, B):
    # define a square matrix in N*N size (N: #nodes)
    data_vector = data_matrix.reshape(num_people*num_people).T
    # given alpha vector, this loop generate dirichlet distribution for N nodes
    pi_list = np.empty(num_people, dtype=object)
    for person in range(num_people):
        person_pi = pymc.Dirichlet('pi_%i' % person, theta=alpha)       
        pi_list[person] = person_pi
    # define two empty matrics in N*N size    
    z_pTq_matrix = np.empty([num_people,num_people], dtype=object)
    z_pFq_matrix = np.empty([num_people,num_people], dtype=object)
    # given dirichlet values, this loop generates multinomial distribution for each mutual nodes
    for p_person in range(num_people):
        for q_person in range(num_people):
            z_pTq_matrix[p_person,q_person] = pymc.Multinomial('z_%dT%d_vector' % (p_person,q_person), n=1, p=pi_list[p_person])
            z_pFq_matrix[p_person,q_person] = pymc.Multinomial('z_%dF%d_vector' % (p_person,q_person), n=1, p=pi_list[q_person])
            
    # matrix B is related to likelihood of relevant contributions or relationship between groups
    # given mutlinomial ditribution and B matrics, this function tries to generate Bernoulli paramteres
    # by dot product of these three matrics
    @pymc.deterministic
    def bernoulli_parameters(z_pTq=z_pTq_matrix, z_pFq=z_pFq_matrix, B=B):
        bernoulli_parameters = np.empty([num_people, num_people], dtype=object)
        for p in range(num_people):
            for q in range(num_people):
                bernoulli_parameters[p,q] = np.dot(np.dot(z_pTq[p,q], B), z_pFq[p,q])
        return bernoulli_parameters.reshape(1,num_people*num_people)
    # compute Bernoulli distribution for mutual nodes   
    y_vector = pymc.Bernoulli('y_vector', p=bernoulli_parameters, value=data_vector, observed=True)
    # return all variables of this function
    return locals()
# load observed graph (network)
data_matrix=np.loadtxt(".../Y_alpha0.1_K2_N20.txt", delimiter=',')
# number of nodes
num_people = 20
# number of block models
num_groups = 2
# initial alpha vector
alpha = np.ones(num_groups).ravel()*0.1
# compute B matrix as symmatric 
B = np.eye(num_groups)*0.8
B = B + np.ones([num_groups,num_groups])*0.2-np.eye(num_groups)*0.2

raw_model = create_model(data_matrix, num_people, num_groups, alpha, B)
print '---------- Finished Running MAP to Set mean-field Variatinal Inference Initial Values ----------'

# mean-field variation inference
# figure 5-page 1990
#==============================================================================
def variational_inference(Y, N, K, B, pi, phi_pqg, phi_qph):
    phi_pqg_hat = np.zeros([num_people, num_people, num_groups])
    phi_qph_hat = np.zeros([num_people, num_people, num_groups])
    # Equation 2 and 3-page 1989
    for p, q in itertools.product(range(N), range(N)):
        inner_fun_output = inner_fun(Y, K, pi, p, q, B, phi_pqg, phi_qph)
        phi_pqg_hat[p, q, :] = inner_fun_output['phi_pg']
        phi_qph_hat[q, p, :] = inner_fun_output['phi_qh']
    return locals()
    
def inner_fun(Y, K, pi, p, q, B, phi_pqg, phi_qph):
    for g in range(K):
        e_term = np.exp(np.mean(np.log(pi[p,g])))
        mult_h = mult_g = 1
        mult1 = (np.power(B[g, :],Y[p, q])*np.power((1-B[g, :]), (1-Y[p, q])))
        mult_h *= np.power(mult1, phi_qph[q, :])
        phi_pg = np.exp((e_term * mult_h) - logsumexp(e_term * mult_h))
        mult2 = (np.power(B[:, g],Y[p, q])*np.power((1-B[:, g]), (1-Y[p, q])))
        mult_g *= np.power(mult2, phi_pqg[p, :])
        phi_qh = np.exp((e_term * mult_g) - logsumexp(e_term * mult_g))
    return locals()
                
#==============================================================================
gamma_pk = np.ones([num_people, num_groups])*((2*num_people)/num_groups)
phi_pqk = np.ones([num_people, num_groups])
phi_qpk = np.ones([num_people, num_groups])
phi_pqg = phi_qph = np.ones([num_people, num_groups])*(1/num_groups)
obs_graph = (1*raw_model['y_vector']._value).reshape(num_people, num_people)
# pymc.Dirichlet generates K-1 variables so that the last one is deterministic
# we define pi_matrix and fill with completed dirichlet values
pi_matrix = np.empty([num_people, num_groups])
dirich_list = list()
for p in range(num_people):
    dirich_list.append(raw_model['pi_list'].item(p)._value.tolist())
    dirich_list[p].append((1-sum(raw_model['pi_list'].item(p)._value)))
pi_matrix = np.array(dirich_list)

VI_outputs = variational_inference(obs_graph, num_people, num_groups, B, pi_matrix, phi_pqg, phi_qph)
 
# Equation 4-page 1989 
phi_pqk = np.array(VI_outputs['phi_pqg_hat'])
phi_qpk = np.array(VI_outputs['phi_qph_hat'])
for p, q in zip(range(num_people), range(num_people)):
    gamma_pk[p, :] = alpha + sum(phi_pqk[p, :, :]) + sum(phi_qpk[:, q, :])

# using Newton-Raphson method to optimize alpha   
# First equation in section 3.2-page 1991
def maximize_alpha(alpha, gamma_pk):
    from numpy.linalg import norm
    from scipy.special import digamma, polygamma
    trigamma = lambda x: polygamma(1, x)
    new_alpha = np.log(alpha)
    n = num_people
    # Second part of gradient, keep this out of alpha_step() because
    # it does not change
    g2 = sum([digamma(gamma_pk[p, :]) - digamma(sum(gamma_pk[p, :])) for p in range(num_people)])
    def alpha_step(alpha): ## Single step of Newton-Rhapson algorithm
        ## From Appendix A.2 of http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf
        old_alpha = alpha
        z = n * trigamma(sum(alpha))
        h = trigamma(alpha)
        ## This part of the gradient changes, update at each step
        g1 = n * (digamma(sum(alpha)) - digamma(alpha))
        g = g1 + g2
        c = sum(g / h) / (1./z + sum(1./h))
        alpha_change = (g - c) / h
        ## Update new_alpha with the Newton-Rhapson update
        new_alpha = alpha + alpha_change
        ## Return how much alpha changed at this step; if it's a big
        ## step, continue running. Otherwise, we can assume we are
        ## close to a local minimum.
        return (new_alpha, norm(new_alpha - old_alpha))
    num_iter = 0
    step_size = 100.0
    ## Run Newton-Rhapson algorithm until convergence, or stop after
    ## some number of iterations.
    while num_iter < 10 and step_size > 0.1:
        (new_alpha, step_size) = alpha_step(new_alpha)
        num_iter += 1
        print num_iter
    return np.exp(new_alpha)

# compute the rho_hat value
# Third equation-page 1991
r1 = np.zeros([num_people, num_people])
inv_obs_graph = 1 - obs_graph
for p, q in itertools.product(range(num_people), range(num_people)):
    r1[p, q] = sum(phi_pqg[p, :] * phi_qph[q, :])
r2 = inv_obs_graph * r1
rho_hat = np.sum(r2)/np.sum(r1)
# optimize beta
# Second equation-page 1991
beta_hat = np.zeros([num_groups, num_groups])
for g, h in itertools.product(range(num_groups), range(num_groups)):
    b1 = sum(obs_graph * (phi_pqg[:, g] * phi_qph[:, h]))
    b2 = (1 - rho_hat) * sum(phi_pqg[:, g] * phi_qph[:, h])
    beta_hat[g, h] = sum(b1)/b2  

print '-------Done!-------'