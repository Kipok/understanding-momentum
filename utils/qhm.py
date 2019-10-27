"""This file contains implementation of qhm algorithm and 
utilities to perform numerical estimation of it's local 
convergence rate"""

import os
import argparse
import numpy as np


def qhm(w, alpha, beta, nu, f, g, T=10000, avg_sz=10000):
    w_cur = w.copy()
    ws = np.empty((T + 1, 2))
    d = np.zeros(2)
    f_val = 0
    for i in range(T):
        ws[i] = w_cur.copy()
        gk = g(w_cur)
        d = beta * d + (1.0 - beta) * gk
        w_cur -= alpha * ((1 - nu) * gk + nu * d)
        if i >= T - avg_sz:
            f_val += f(w_cur)
    f_val = f_val / avg_sz
    ws[T] = w_cur.copy()
    return ws, f_val


def qhm_rate(alpha, beta, nu, lambds):
    rate = 0.0
    for l in lambds:
        c1 = 1 - alpha * l + alpha * nu * beta * l + beta
        c2 = beta * (1 - alpha * l + alpha * nu * l)
        D = c1 ** 2 - 4 * c2
        if D >= 0:
            lamb = max(np.abs(c1 + np.sqrt(D)) / 2, np.abs(c1 - np.sqrt(D)) / 2)
        else:
            lamb = np.sqrt(c2)
        rate = max(rate, lamb)
    return rate


def regime(alpha, beta, nu, l):
    c1 = 1 - alpha * l + alpha * nu * beta * l + beta
    c2 = beta * (1 - alpha * l + alpha * nu * l)
    if c1 >= 0 and c1 ** 2 - 4 * c2 >= 0:
        return 1
    if c1 < 0 and c1 ** 2 - 4 * c2 >= 0:
        return 2
    if c1 ** 2 - 4 * c2 < 0:
        return 3

    
def qhm_rate_split(alpha, beta, nu, l):
    c1 = 1 - alpha * l + alpha * nu * beta * l + beta
    c2 = beta * (1 - alpha * l + alpha * nu * l)
    D = c1 ** 2 - 4 * c2
    if D >= 0 and c1 >= 0:
        lamb = (c1 + np.sqrt(D)) / 2
    elif D >= 0 and c1 < 0:
        lamb = (-c1 + np.sqrt(D)) / 2
    elif D < 0:
        lamb = np.sqrt(c2)
    return lamb


def alpha_solver(beta, nu, mu, L, eps=1e-8):
    def solver(alpha, l):
        c1 = 1 - alpha * l + alpha * l * nu * beta + beta
        c2 = beta * (1 - alpha * l + alpha * l * nu)

        if regime(alpha, beta, nu, l) == 3:
            return c2 ** 0.5
        else:
            return 0.5 * (np.abs(c1) + np.sqrt(c1 ** 2 - 4 * c2))
    
    def lhs(alpha):
        return solver(alpha, mu)
    
    def rhs(alpha):
        return solver(alpha, L)
    
    alpha_left = 0
    alpha_right = 2 * (1 + beta) / (L * (1 + beta * (1 - 2 * nu)))
    while alpha_right - alpha_left > eps:
        alpha = (alpha_left + alpha_right) / 2
        if lhs(alpha) > rhs(alpha):
            alpha_left = alpha
        else:
            alpha_right = alpha
    return (alpha_left + alpha_right) / 2


def alpha_beta_solver(nu, mu, L, grid_size=1000, alpha_eps=1e-8):
    betas = np.linspace(0.0, 1.0 - 1e-5, grid_size)
    res = np.empty(grid_size)
    for i, beta in enumerate(betas):
        res[i] = qhm_rate(alpha_solver(beta, nu, mu, L, alpha_eps), beta, nu, [mu, L])
    opt_beta = betas[np.argmin(res)]
    opt_alpha = alpha_solver(opt_beta, nu, mu, L, alpha_eps)
    return opt_alpha, opt_beta


def beta_solver(nu, mu, L, sz=1000):
    betas = np.linspace(0.0, 0.999, sz)
    res = np.empty(sz)
    for i, beta in enumerate(betas):
        res[i] = qhm_rate(alpha_solver(beta, nu, mu, L), beta, nu, [mu, L])
    opt_beta = betas[np.argmin(res)]
    opt_alpha = alpha_solver(beta, nu, mu, L)
    return opt_beta


def nu_solver(beta, mu, L, sz=1000):
    nus = np.linspace(0.0, 1.0, sz)
    res = np.empty(sz)
    for i, nu in enumerate(nus):
        res[i] = qhm_rate(alpha_solver(beta, nu, mu, L), beta, nu, [mu, L])
    opt_nu = nus[np.argmin(res)]
    opt_alpha = alpha_solver(beta, nu, mu, L)
    return opt_nu
