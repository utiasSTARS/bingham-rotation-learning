import numpy as np
from scipy.stats import chi2
from numpy.linalg import norm 
from liegroups.numpy import SO3
import matplotlib.pylab as plt
import cvxpy as cp
import time
from helpers import *

def build_A(x_1, x_2, sigma_2):
    N = x_1.shape[0]
    A = np.zeros((4, 4))
    for i in range(N):
        # Block diagonal indices
        I = np.eye(4)
        t1 = (x_2[i].dot(x_2[i]) + x_1[i].dot(x_1[i]))*I
        t2 = 2*Omega_l(pure_quat(x_2[i])).dot(
            Omega_r(pure_quat(x_1[i])))
        A_i = (t1 + t2)/(sigma_2[i])
        A += A_i
    return A   

def solve_wahba(A, redundant_constraints=False):
    #Input: x_1, x_2: N x 3 matrices, sigma_2_i is N dimensional
    start = time.time()

    # Build Q variable with constraints
    Q = cp.Variable((4, 4), PSD=True)

    # Naive constraints
    constraints = [cp.trace(Q) == 1]

    # Additional non-naive constraints
    if redundant_constraints:
        constraints += [Q == Q.T]

    # Solve SDP
    prob = cp.Problem(cp.Minimize(cp.trace(A @ Q)),
                      constraints)
    prob.solve(solver=cp.CVXOPT, verbose=False)
    t_solve = time.time() - start #Note: the following only seems to properly work with MOSEK: prob.solver_stats.solve_time
    # eigs = np.linalg.eigvals(Q.value)
    # rank_Q = np.sum(eigs > rank_tol)
    q_est = q_from_qqT(Q.value)
    primal_cost = np.dot(q_est.T, np.dot(A, q_est))
    gap = primal_cost - prob.solution.opt_val[0]
    nu = constraints[0].dual_value

    return q_est, nu, t_solve, gap 


def compute_grad(A, nu, q):
    #Returns 4x4x4 gradient tensor G where G[:,i,j] is dq/dA_ij
    G = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            G[:, i, j] = compute_grad_ij(A, nu, q, i, j)
    return G

def compute_grad_ij(A, nu, q, i, j):
    #Computes 4x1 gradient, dq/dA_ij where A_ij is A[i,j]s
    I_ij = np.zeros((4,4))
    I_ij[i,j] = 1
    # This formulation runs into numerical stability issues
    # M = np.linalg.inv(A + nu*np.eye(4))
    # term = M.dot(q.T.dot(q)).dot(M) / (q.dot(A).dot(q.T))
    # grad = -(M - term).dot(I_ij).dot(q.T)

    M = np.zeros((5,5))
    M[:4,:4] = A + nu*np.eye(4)
    M[4,:4] = q
    M[:4,4] = q.T

    b = np.zeros(5)
    b[:4] = I_ij.dot(q)
    dz = - np.linalg.solve(M, -1*b)
    grad = dz[:4]
    
    return grad

def gen_sim_data(N=100, sigma=0.01):
    ##Simulation
    #Create a random rotation
    C = SO3.exp(np.random.randn(3)).as_matrix()
    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = normalized(np.random.rand(N, 3) - 0.5, axis=1)
    #Rotate and add noise
    x_2 = C.dot(x_1.T).T + sigma*np.random.randn(N,3)

    return C, x_1, x_2


def check_gradients():
    N = 10
    sigma = 1

    print('Checking gradients...')
    _, x_1, x_2 = gen_sim_data(N, sigma)
    A = build_A(x_1, x_2, sigma*sigma*np.ones(N))
    q_opt, nu_opt, _, _ = solve_wahba(A, redundant_constraints=True)
    G_analytic = compute_grad(A, nu_opt, q_opt)
    

    G_numerical = np.zeros((4, 4, 4))
    dth = 1e-6
    print('Using dA_ij = {:.3E}.'.format(dth))
    for i in range(4):
        for j in range(4):
            dA_ij = np.zeros((4,4))
            dA_ij[i,j] = dth
            q_plus,_,_,_ =  solve_wahba(A+dA_ij, redundant_constraints=True)
            q_minus,_,_,_ =  solve_wahba(A-dA_ij, redundant_constraints=True)
            G_numerical[:,i,j] = (q_plus - q_minus)/(2.*dth)

    norm_diff = np.linalg.norm(G_analytic - G_numerical)
    print('Matrix norm difference of numerical vs analytic: {:.5f}'.format(norm_diff))

    return

def check_single_solve():
    N = 100
    sigma = 0.01
    C, x_1, x_2 = gen_sim_data(N, sigma)
    redundant_constraints = False
    ## Solver
    A = build_A(x_1, x_2, sigma*sigma*np.ones(N))
    q_est, nu, t_solve, gap = solve_wahba(A, redundant_constraints=redundant_constraints)
    C_est = SO3.from_quaternion(q_est, ordering='xyzw').as_matrix()
    print('Done. Solved in {:.3f} seconds.'.format(t_solve))
    print('Duality gap: {:.3E}.'.format(gap))
    #Compare to known rotation
    print('Convex rotation error: {:.3f} deg'.format(so3_error(C_est, C)))
    print('Horn rotation error: {:.3f} deg'.format(so3_error(solve_horn(x_1, x_2), C)))

if __name__=='__main__':
    #check_single_solve()
    check_gradients()
    


