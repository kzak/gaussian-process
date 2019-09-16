# coding: utf-8

import numpy as np


def kernel_func(xi, xj, params):
    e_tau, e_sigma, e_eta = params

    # dist_ij = |xi - xj|^2
    dist_ij = (xi - xj) ** 2

    # delta(xi, xj)
    delta_ij = (xi == xj)

    # k(xi, xj | theta), theta = {tau, sigma, eta}
    k_ij = e_tau * np.exp(- (dist_ij / e_sigma)) + e_eta * delta_ij

    return k_ij


class GP:
    def __init__(self, Y, X, kernel_func, params, params_ranges):
        self.Y = Y
        self.X = X
        self.kernel_func = kernel_func
        self.params = [np.exp(p) for p in params]
        self.params_ranges = params_ranges

    # train the model (without hyper parameter optimization)
    def train(self):
        self.K00 = self.kernel_func(
            *np.meshgrid(self.X, self.X), self.params)

        self.K00_inv = np.linalg.inv(self.K00)

    # predict y (=mu, std) from x
    def predict(self, x):
        K00_inv = self.K00_inv
        K01 = self.kernel_func(*np.meshgrid(self.X, x, indexing='ij'), self.params)
        K10 = K01.T
        K11 = self.kernel_func(*np.meshgrid(x, x), self.params)

        mu = K10.dot(K00_inv.dot(self.Y))
        sigma = K11 - K10.dot(K00_inv.dot(K01))
        std = np.sqrt(sigma.diagonal())
        return mu, std

    def loglik(self, params=None):
        if params is None:
            params = self.params
        
        K00 = self.kernel_func(*np.meshgrid(self.X, self.X), params)
        K00_inv = np.linalg.inv(K00)
        
        # see about np.linalg.slogdet
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.slogdet.html
        return -(np.linalg.slogdet(K00)[1] + self.Y.dot(K00_inv.dot(self.Y)))

    def optimize_mcmc(self, n_iter=1000, lr=0.1):
        n_params = len(self.params)
        s = (self.params_ranges[:, 1] - self.params_ranges[:, 0])

        params_prev = self.params
        ll_prev = self.loglik(params_prev)
        
        params_list = []
        ll_list = []
        
        for i in range(n_iter):
            deltas = np.random.normal(0, s, n_params)

            params_next = params_prev + lr * deltas
            ll_next = self.loglik(params_next)

            r = np.exp(ll_next - ll_prev)
            if(r >= 1 or r > np.random.random()):
                params_list.append(params_next)
                ll_list.append(ll_next)

                params_prev = params_next
                ll_prev = ll_next
                
        # Update kernal parameters
        self.params = np.exp(params_list[np.argmax(ll_list)])
        
        # and retrain the model.
        self.K00 = kernel_func(*np.meshgrid(self.X, self.X), self.params)
        self.K00_inv = np.linalg.inv(self.K00)
    