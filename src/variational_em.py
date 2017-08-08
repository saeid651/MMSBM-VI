################################################################
##
## VARIATIONAL EM (VEM) FOR PARAMETER ESTIMATION OF MMSBMs
##
## Saeid Parvandeh and Chad Crawford
##
################################################################

import numpy as np
from scipy.misc import logsumexp
from scipy.special import digamma, polygamma

trigamma = lambda x: polygamma(1, x)

class VariationalEM:
    def __init__(self, num_people, num_groups, num_iterations=20):
        # Model properties
        self.num_people = num_people
        self.num_groups = num_groups

        # Algorithm parameters
        self.num_iterations = num_iterations

        # Initialize parameters
        self.gamma = np.ones([num_people, num_groups]) * ((2*num_people)/num_groups)
        self.phi_to = np.ones([num_people, num_people, num_groups])
        self.phi_from = np.ones([num_people, num_people, num_groups])

        self.alpha = np.ones([num_people, num_groups])
        self.b = np.ones([num_groups, num_groups])

    def estimate(self, y):
        for _ in range(self.num_iterations):
            self.e_step(data)
            self.m_step(data)

    def e_step(self, y):
        # Coordinate ascent to solve for phi_to, phi_from & gamma
        for _ in range(10):
            self.vi_phi_to(y)
            self.vi_phi_from(y)
            self.vi_gamma(y)

    def vi_phi_to(self, y, e_term):
        # Maximize phi_to
        for p in range(self.num_people):
            t1 = digamma(self.gamma[p, :]) - digamma(sum(self.gamma[p, :]))
            for q in range(self.num_people):
                t2 = self.phi_from[p, q, :] * sum([
                    y[p, q] * np.log(self.b[g, :]) \
                    + (1-y[p, q]) * np.log(1-self.b[g, :])
                    for g in range(self.num_groups)]
                )
                t = t1 + t2
                # Normalize over groups
                self.phi_to[p, q, :] = np.exp(t - logsumexp(t))

    def vi_phi_from(self, y):
        for q in range(self.num_people):
            t1 = digamma(self.gamma[q, :]) - digamma(sum(self.gamma[q, :]))
            for p in range(self.num_people):
                t2 = self.phi_from[p, q, :] * sum([
                    y[p, q] * np.log(self.b[:, h]) \
                    + (1-y[p, q]) * np.log(1-self.b[:, h])
                    for h in range(self.num_groups)]
                )
                t = t1 + t2
                self.phi_from[p, q, :] = np.exp(t - logsumexp(t))

    def vi_gamma(self, y):
        for p in range(self.num_people):
            self.gamma[p, :] = self.alpha \
                + sum(self.phi_to[p, q, :] for q in range(self.num_people)) \
                + sum(self.phi_from(p, q) for q in range(self.num_people))

    def m_step(self, y):
        self.max_alpha(y)
        self.max_b(y)

    def max_alpha(self, y):
        from numpy.linalg import norm
        new_alpha = self.alpha
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
        self.alpha = new_alpha

    def max_b(self, y):
        r1 = np.zeros([num_people, num_people])
        inv_obs_graph = 1 - obs_graph
        for p, q in itertools.product(range(num_people), range(num_people)):
            r1[p, q] = sum(phi_pqg[p, :] * phi_qph[q, :])
        r2 = inv_obs_graph * r1
        rho_hat = np.sum(r2)/np.sum(r1)

        for g, h in itertools.product(range(num_groups), range(num_groups)):
            b1 = sum(obs_graph * (phi_pqg[:, g] * phi_qph[:, h]))
            b2 = (1 - rho_hat) * sum(phi_pqg[:, g] * phi_qph[:, h])
            self.beta[g, h] = sum(b1)/b2
