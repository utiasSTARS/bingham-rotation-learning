import numpy as np
import scipy as sp
from liegroups.numpy import SO3
import cvxpy as cp
import time
from helpers import *
import torch

def build_A(x_1, x_2, sigma_2):
    N = x_1.shape[0]
    A = np.zeros((4, 4), dtype=np.float64)
    for i in range(N):
        # Block diagonal indices
        I = np.eye(4, dtype=np.float64)
        t1 = (x_2[i].dot(x_2[i]) + x_1[i].dot(x_1[i]))*I
        t2 = 2.*Omega_l(pure_quat(x_2[i])).dot(
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
    if Q.value is None:
        print('Q is empty. Solver may have failed.')
        return
    q_opt = q_from_qqT(Q.value)
    primal_cost = np.dot(q_opt.T, np.dot(A, q_opt))
    gap = primal_cost - prob.solution.opt_val[0]
    nu_opt = constraints[0].dual_value
    return q_opt, nu_opt, t_solve, gap 


def solve_wahba_fast(A, redundant_constraints=False):
    """
    Use a fast eigenvalue solution to the dual of the 'generalized Wahba' problem to solve the primal.
    :param A: quadratic cost matrix
    :param redundant_constraints: boolean indicating whether to use redundand constraints
    :return: Optimal q, optimal dual var. nu, time to solve, duality gap
    """
    start = time.time()
    nus, qs = torch.eig(A, eigenvectors=True)
    print('Nus: {:}'.format(nus))
    print('qs: {:}'.format(qs))
    real_ids = nus[:, 1] == 0
    print('Real_ids: {:}'.format(real_ids))
    print(nus[real_ids, 0])
    nus = nus[real_ids, 0]
    qs = qs[:, real_ids]
    # id_min = torch.argmin(nus[:, 0]) # Only check real part
    id_min = torch.argmin(nus)
    nu_opt = -nus[id_min]
    q = -qs[:, id_min]
    q_opt = q/torch.norm(q, 2)
    t_solve = time.time() - start
    # gap = torch.einsum('bnk,bkl,bnl->b', q.unsqueeze(0), A, q.unsqueeze(0)) - nu_opt
    gap = q@A@q - nu_opt
    return q_opt, nu_opt, t_solve, gap


def compute_grad(A, nu, q):
    #Returns 4x4x4 gradient tensor G where G[:,i,j] is dq/dA_ij
    G = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            G[:, i, j] = compute_grad_ij(A, nu, q, i, j)
    return G


def compute_grad_ij(A, nu, q, i, j):
    #Computes 4x1 gradient, dq/dA_ij where A_ij is A[i,j]s

    I_ij_ji = np.zeros((4,4))
    I_ij_ji[i,j] += 1
    I_ij_ji[j,i] += 1
    # This formulation runs into numerical stability issues
    # M = np.linalg.inv(A + nu*np.eye(4))
    # term = M.dot(q.T.dot(q)).dot(M) / (q.dot(A).dot(q.T))
    # grad = -(M - term).dot(I_ij).dot(q.T)

    M = np.zeros((5,5))
    M[:4,:4] = A + A.T + 2*nu*np.eye(4)
    M[4,:4] = 2*q
    M[:4,4] = 2*q.T

    b = np.zeros(5)
    b[:4] = I_ij_ji.dot(q)
    dz = -1*sp.linalg.solve(M, b)
    grad = dz[:4]
    
    return grad