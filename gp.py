# coding: utf-8

import numpy as np


def kernel_func(xi, xj, params):
    tau, sigma, eta = params
    e_tau, e_sigma, e_eta = np.exp(params)

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
        self.params = params
        self.params_ranges = params_ranges

    # train the model (without hyper parameter optimization)
    def train(self):
        self.K00 = self.kernel_func(*np.meshgrid(self.X, self.X), self.params)
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

#            print(i, ll_next, params_next)
            r = np.exp(ll_next - ll_prev)
            if(r >= 1 or r > np.random.random()):
                params_list.append(params_next)
                ll_list.append(ll_next)

                params_prev = params_next
                ll_prev = ll_next
                
        # Update kernal parameters
        # import pdb; pdb.set_trace()
        self.params = params_list[np.argmax(ll_list)]
        
        # and retrain the model.
        self.K00 = kernel_func(*np.meshgrid(self.X, self.X), self.params)
        self.K00_inv = np.linalg.inv(self.K00)

    def loglik_grads(self, params):
        tau, sigma, eta = params
        e_tau, e_sigma, e_eta = np.exp(params)

        X = self.X.reshape(len(self.X), 1)
        Y = self.Y

        K = self.K00
        K_inv = np.linalg.inv(K)
        K_inv_Y = np.dot(K_inv, Y)
        D = np.eye(len(K))

        X = self.X.reshape(len(K), 1)
        dist_X = np.linalg.norm(X[:, np.newaxis] - X, axis=-1)

        K_tau = self.K00 - e_eta * D
        K_sigma = np.dot((self.K00 - e_eta * D), (np.exp(-sigma)) * dist_X)
        K_eta = e_eta * D

#        import pdb; pdb.set_trace()

        grad_tau = - np.trace(np.dot(K_inv, K_tau)) + \
            np.dot(K_inv_Y.T, K_tau).dot(K_inv_Y)

        grad_sigma = - np.trace(np.dot(K_inv, K_sigma)) + \
            np.dot(K_inv_Y.T, K_sigma).dot(K_inv_Y)

        grad_eta = - np.trace(np.dot(K_inv, K_eta)) + \
            np.dot(K_inv_Y.T, K_eta).dot(K_inv_Y)

        return np.array([grad_tau, grad_sigma, grad_eta])

    def optimize_grads(self, n_iter=1000, lr=0.1):
        params = self.params.copy()
        tau, sigma, eta = params

        for i in range(n_iter):
            
            delta_tau, delta_sigma, delta_eta = self.loglik_grads(params)

            tau = tau + lr * delta_tau
            sigma = sigma + lr * delta_sigma
            eta = eta + lr * delta_eta

            params = np.array([tau, sigma, eta])
            self.params = params
            self.train()

        self.params = params
