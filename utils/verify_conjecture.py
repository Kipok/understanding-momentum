import os
import argparse
import numpy as np
from qhm import alpha_beta_solver, qhm_rate


def verify_conjecture(args):
    if args.dump_results:
        os.mkdir(args.dump_results)
    kappas = np.linspace(args.kappa_lb, args.kappa_rb, args.kappa_grid_size)
    conj_verified = True
    for kappa in kappas:
        mu = 1.0
        L = kappa
        # doing this since mu = L is invalid for alpha_solver
        if kappa == 1.0:
            L += 1e-5
        nus = np.linspace(0.0, 1.0, args.nu_grid_size)
        rates_nu = np.empty(nus.shape[0])
    
        for i, nu in enumerate(nus):
            opt_alpha, opt_beta = alpha_beta_solver(
                nu, mu, L, grid_size=int(1.0 / args.beta_eps), 
                alpha_eps=args.alpha_eps,
            )
            rates_nu[i] = qhm_rate(opt_alpha, opt_beta, nu, [mu, L])
        if args.dump_results:
            np.save(os.path.join(args.dump_results, 'kappa-{}.npy'.format(kappa)), rates_nu)
        # checking that every 10-th nu is approximately monotonically decreasing
        # to counteract numerical ripples on the rate plot
        for st in range(10):
            if not np.alltrue(np.diff(rates_nu[st::10]) <= 1e-3):
                print("Conjecture disproved for kappa = {}".format(kappa))
                conj_verified = False
                break
    if conj_verified:
        print("Conjecture verified!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check QHM rate')
    parser.add_argument('--alpha_eps', default=1e-8, type=float,
                        help='Precision on estimating alpha')
    parser.add_argument('--beta_eps', default=1e-3, type=float,
                        help='Precision on estimating beta')
    parser.add_argument('--nu_grid_size', default=1000, type=int,
                        help='Grid size for nu')
    parser.add_argument('--kappa_grid_size', default=1000, type=int,
                        help='Grid size for kappa')
    parser.add_argument('--kappa_lb', default=1.0, type=float,
                        help='Left bound for kappa grid')
    parser.add_argument('--kappa_rb', default=1e5, type=float,
                        help='Right bound for kappa grid')
    parser.add_argument('--dump_results', type=str, default="",
                        help='Dump rates estimation to this folder')
    args = parser.parse_args() 

    verify_conjecture(args)

