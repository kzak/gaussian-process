# coding: utf-8

import pathmagic # noqa
import pytest

from gp import * # noqa

import numpy as np


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
    return [1.0, 1.0, 1.0]


@pytest.fixture
def params_ranges():
    return np.array([[1e-2, 1], [1e-2, 1], [1e-2, 1]])


@pytest.fixture
def gp(Y, X, params, params_ranges):
    gp = GP(Y, X, kernel_func, params, params_ranges)
    gp.train()
    return gp


##############################
# Tests
##############################

def test_kernel_func(X, params):
    kernel_func(*np.meshgrid(X, X), params)


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


def test_GP_optimize_mcmc(gp):
    params_prev = gp.params

    gp.optimize_mcmc(n_iter=10)

    params_next = gp.params

    assert all(params_next != params_prev)
