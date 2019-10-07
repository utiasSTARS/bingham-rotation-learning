import numpy as np
from scipy.stats import chi2
from numpy.linalg import norm 
from liegroups.numpy import SO3
import matplotlib.pylab as plt
import cvxpy as cp
import time
from helpers import *


def solve_quasar(x_1, x_2, c_bar_2, redundant_constraints=False, rank_tol=1e-4):
    N = x_1.shape[0]
    Q = np.zeros((4 * (N + 1), 4 * (N + 1)))
    sigma_2_i = 1  # Incorporate sigma into c_bar_2
    # for i in range(N):
    for ii in range(N):
        Q_i = np.zeros((4 * (N + 1), 4 * (N + 1)))
        # Block diagonal indices
        idx_range = slice((ii + 1) * 4, (ii + 2) * 4)
        Q_i[idx_range, idx_range] = Q_ii(x_1[ii], x_2[ii], c_bar_2, sigma_2_i)
        Q_0ii = Q_0i(x_1[ii], x_2[ii], c_bar_2, sigma_2_i)
        Q_i[:4, idx_range] = Q_0ii
        Q_i[idx_range, :4] = Q_0ii
        Q += Q_i

    # Build Z variable with constraints
    Z = cp.Variable((4 * (N + 1), 4 * (N + 1)), PSD=True)

    # Naive constraints
    constraints = [
        cp.trace(Z[:4, :4]) == 1
    ]
    constraints += [
        Z[(i) * 4:(i + 1) * 4, (i) * 4:(i + 1) * 4] == Z[:4, :4] for i in range(1, N + 1)
    ]

    # Additional non-naive constraints
    if redundant_constraints:
        # q q_i
        constraints += [
            Z[:4, (i) * 4:(i + 1) * 4] == Z[:4, (i) * 4:(i + 1) * 4].T for i in range(1, N + 1)
        ]
        # q_i q_j
        for i in range(1, N + 1):
            constraints += [
                Z[4 * i:4 * (i + 1), (j) * 4:(j + 1) * 4] == Z[4 * i:4 * (i + 1), (j) * 4:(j + 1) * 4].T for j in
                range(i + 1, N + 1)
            ]

    # Solve SDP
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ Z)),
                      constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    t_solve = prob.solver_stats.solve_time
    eigs = np.linalg.eigvals(Z.value)
    rank_Z = np.sum(eigs > rank_tol)
    q_est = q_from_qqT(Z.value[:4, :4])
    # Extract outliers
    est_outlier_indices, est_inlier_indices = extract_outlier_indices(Z.value)
    q_full_est = np.zeros((4*(N+1), 1))
    q_full_est[0:4, 0] = q_est
    for idx in range(1, N+1):
        if idx-1 in est_outlier_indices:
            q_full_est[4*idx:4*(idx+1), 0] = -q_est
        else:
            q_full_est[4 * idx:4 * (idx + 1), 0] = q_est
    primal_cost = np.dot(q_full_est.T, np.dot(Q, q_full_est))
    gap = primal_cost - prob.solution.opt_val
    return q_est, list(est_outlier_indices), t_solve, gap, prob.solution.opt_val

if __name__=='__main__':
    ##Parameters
    #Sim
    N = 10
    sigma = 0.01 #0.01
    N_out = 5  # How many of N samples are outliers

    #Solver
    # sigma_2_i = sigma**2
    #p_false_negative = 0.001 # Probability an inlier is classified as an outlier
    #c_bar_2 = chi2.ppf(1-p_false_negative, df=3)

    #SDP seems to react poorly to small values of sigma_2_i
    #Instead, we rely on the interpretation that we reject any residual larger than sigma_i*c_bar as an outlier
    #Set sigma_i to 1, set c_bar to 3*sigma
    sigma_2_i = 1
    c_bar_2 = (3*(sigma+1e-3))**2
    print('c_bar_2: {:.3f}'.format(c_bar_2))
    redundant_constraints = True

    ##Simulation
    #Create a random rotation
    C = SO3.exp(np.random.randn(3)).as_matrix()

    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = normalized(np.random.rand(N, 3) - 0.5, axis=1)
    #Rotate and add noise
    x_2 = C.dot(x_1.T).T + sigma*np.random.randn(N,3)

    #Outliers
    if N_out > 0:
        outlier_indices = np.random.choice(x_2.shape[0], N_out, replace=False)
        x_2[outlier_indices] = 10*(np.random.rand(N_out, 3) - 0.5)

    ## Solver
    #Build Q matrix
    #No sparsity for now
    q_est, est_outlier_indices, t_solve, gap = solve_quasar(x_1, x_2, c_bar_2, redundant_constraints=redundant_constraints)
    #Extract outliers
    # est_outlier_indices = extract_outlier_indices(Z.value)
    outlier_indices = list(outlier_indices)
    outlier_indices.sort()

    C_est = SO3.from_quaternion(q_est, ordering='xyzw').as_matrix()
    print('Done. Solved in {:.3f} seconds.'.format(t_solve))
    # print('Found all outliers: {}'.format(outlier_indices == est_outlier_indices))

    #Compare to known rotation
    # C_est = C
    print('SO(3) Frob norm error:')
    print(np.linalg.norm(C-C_est))
