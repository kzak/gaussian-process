# coding: utf-8

import pathmagic # noqa
import pytest

from gp import * # noqa

import numpy as np
np.random.seed(42)


##############################
# Fixtures
##############################

@pytest.fixture
def N():
    return 50

@pytest.fixture
def X(N):
    return np.random.uniform(0, 100, N)


@pytest.fixture
def Y(X, N):
    return 5 * np.sin(np.pi / 15 * X) * np.exp(-X / 50) + np.random.randn(N)


@pytest.fixture
def params():
    return [3.0, 3.0, 3.0]


@pytest.fixture
def params_ranges():
    return np.array([[1, 5], [1, 5], [1, 5]])


@pytest.fixture
def kernel(params, params_ranges):
    return Kernel(params, params_ranges)


@pytest.fixture
def gp(Y, X, kernel):
    gp = GP(Y, X, kernel)
    gp.train()
    return gp


##############################
# Tests
##############################

def test_Kernel_init(kernel):
    assert isinstance(kernel, Kernel)


def test_Kernel_kernel_matrix(kernel, X):
    K = kernel(X)

    nrow_X = len(X)
    assert (nrow_X, nrow_X) == K.shape


def test_GP_init(gp):
    assert isinstance(gp, GP)


def test_GP_train(gp):
    assert (50, 50) == gp.K00.shape
    assert gp.K00.shape == gp.K00_inv.shape


def test_GP_predict(gp):
    n_xs = 10
    xs = np.linspace(0, 100, n_xs)

    mu, std = gp.predict(xs)

    assert (n_xs, ) == mu.shape
    assert (n_xs, ) == std.shape


def test_GP_loglik(gp):
    ll = gp.loglik()

    assert isinstance(ll, float)


def test_GP_optimize_mcmc(gp, params):
    params_prev = gp.kernel.params

    gp.optimize_mcmc(n_iter=10)

    params_next = gp.kernel.params

    assert all(params_next != params_prev)


# def test_GP_loglik_grads(gp, params):
#     grads = gp.loglik_grads(params)

#     assert (3, ) == grads.shape


# def test_GP_optimize_grads(gp, params):
#     params_prev = gp.kernel.params

#     gp.optimize_grads(n_iter=100)

#     params_next = gp.kernel.params

#     assert all(params_next != params_prev)



    
