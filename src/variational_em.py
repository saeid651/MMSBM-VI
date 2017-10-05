#!/usr/bin/env python3

################################################################
##
## VARIATIONAL EM (VEM) FOR PARAMETER ESTIMATION OF MMSBMs
##
## Saeid Parvandeh and Chad Crawford
##
################################################################

import numpy as np
import numpy.random as npr
import itertools
from scipy.misc import logsumexp
from scipy.special import digamma, polygamma

trigamma = lambda x: polygamma(1, x)

class VariationalEM:
    def __init__(self, num_people, num_groups, num_iterations=20, learn_sparsity=True):
        # Model properties
        self.num_people = num_people
        self.num_groups = num_groups

        # Algorithm parameters
        self.num_iterations = num_iterations

        self.alpha = npr.rand(num_groups) / 2
        self.b = np.eye(num_groups) + 1e-4
        for g in range(num_groups):
            self.b[g, :] = self.b[g, :] / np.sum(self.b[g, :])
        self.rho = 0.1

        # Initialize parameters
        self.gamma = npr.rand(num_people, num_groups) / 2.
        self.phi_to = np.ones([num_people, num_people, num_groups]) / num_groups
        self.phi_from = np.ones([num_people, num_people, num_groups]) / num_groups

        self.learn_sparsity = learn_sparsity

    def train(self, y):
        """Parameter estimation method; y is the n x n evidence matrix."""
        if not self.learn_sparsity:
            self.rho = 1 - (np.sum(y) / pow(self.num_people, 2))
        for _ in range(self.num_iterations):
            print "b =", self.b
            print "alpha =", self.alpha
            self.e_step(y)
            self.m_step(y)

    def e_step(self, y):
        """Expectation step of EM algorithm."""
        # Coordinate ascent to solve for phi_to, phi_from & gamma
        for _ in range(10):
            for q in range(self.num_people):
                # t1 = digamma(self.gamma[p, :]) - digamma(sum(self.gamma[p, :]))
                for p in range(self.num_people):
                    self.vi_phi_to(y, p, q)
                    self.vi_phi_from(y, p, q)
                    self.vi_gamma(y, p)
                    self.vi_gamma(y, q)

    def m_step(self, y):
        """Maximization step of EM algorithm."""
        self.max_alpha(y)
        if self.learn_sparsity:
            self.max_rho(y)
        self.max_b(y)

    def vi_phi_to(self, y, p, q):
        """Variational inference for phi_to term."""
        t1 = digamma(self.gamma[p, :]) - digamma(sum(self.gamma[p, :]))
        t2 = y[p, q] * np.log(self.b) + (1-y[p, q]) * np.log(1-self.b)
        t2 = np.sum(t2, axis=1)
        t2 = np.multiply(self.phi_from[p, q, :], t2)
        t = t1 + t2
        # Normalize over groups
        self.phi_to[p, q, :] = np.exp(t - logsumexp(t))

    def vi_phi_from(self, y, p, q):
        """Variational inference for phi_from term."""
        t1 = digamma(self.gamma[q, :]) - digamma(sum(self.gamma[q, :]))
        t2 = y[p, q] * np.log(self.b) + (1-y[p, q]) * np.log(1-self.b)
        t2 = np.sum(t2, axis=1)
        t2 = np.multiply(self.phi_to[p, q, :], t2)
        t = t1 + t2
        self.phi_from[p, q, :] = np.exp(t - logsumexp(t))

    def vi_gamma(self, y, p):
        """Variational inference for gamma term."""
        t2 = np.add.reduce([self.phi_to[p, q, :] for q in range(self.num_people)])
        t3 = np.add.reduce([self.phi_from[p, q, :] for q in range(self.num_people)])
        self.gamma[p, :] = self.alpha + t2 + t3

    def max_alpha(self, y):
        from numpy.linalg import norm
        new_alpha = self.alpha
        n = self.num_people
        # Second part of gradient, keep this out of alpha_step() because
        # it does not change
        g2 = np.add.reduce([digamma(self.gamma[p, :]) - digamma(sum(self.gamma[p, :])) for p in range(n)])
        avg_gamma = np.mean(self.gamma, axis=0)
        def alpha_step(alpha): ## Single step of Newton-Rhapson algorithm
            ## From Appendix A.2 of http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf
            old_alpha = alpha
            z = n * trigamma(sum(alpha))
            h = - n * trigamma(alpha)
            ## This part of the gradient changes, update at each step
            g1 = n * (digamma(sum(alpha)) - digamma(alpha))
            g = g1 + g2
            b = sum(g / h) / ((1./z) + sum(1./h))
            alpha_change = (g - b) / h
            ## Update alpha
            new_alpha = alpha - alpha_change
            # print g1, g2
            # print new_alpha, norm(g), norm(alpha_change)
            ## Return how much alpha changed at this step; if it's a big
            ## step, continue running. Otherwise, we can assume we are
            ## close to a local minimum.
            return (new_alpha, norm(alpha_change))
        num_iter = 0
        step_size = 100.0
        ## Run Newton-Rhapson algorithm until convergence, or stop after
        ## some number of iterations.
        while num_iter < 10 and step_size > 0.01:
            (new_alpha, step_size) = alpha_step(new_alpha)
        num_iter += 1
        self.alpha = new_alpha

    def max_rho(self, y):
        y2 = 1-y
        num = np.zeros((self.num_people, self.num_people))
        denom = np.zeros((self.num_people, self.num_people))
        for g, h in itertools.product(range(self.num_groups), range(self.num_groups)):
            m1 = np.multiply(self.phi_from[:, :, g], self.phi_to[:, :, h])
            num += np.multiply(y2, m1)
            denom += m1
        self.rho = np.sum(num) / np.sum(denom)

    def max_b(self, y):
        for g, h in itertools.product(range(self.num_groups), range(self.num_groups)):
            tmp_m = np.multiply(self.phi_from[:, :, g], self.phi_to[:, :, h])
            t1 = np.sum(np.multiply(y, tmp_m))
            t2 = (1 - self.rho) * np.sum(tmp_m)
            self.b[g, h] = (t1 / t2) + 1e-4
        ## Normalization
        for g in range(self.num_groups):
            self.b[g, :] = self.b[g, :] / np.sum(self.b[g, :])
