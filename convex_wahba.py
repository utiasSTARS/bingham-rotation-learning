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
    # Returns (b,n) and (b,n,n) tensors
    nus, qs = torch.symeig(A, eigenvectors=True)
    nu_min, nu_argmin = torch.min(nus, 1)# , keepdim=False, out=None)
    q_opt = qs[torch.arange(A.shape[0]), :, nu_argmin]
    q_opt = q_opt*(torch.sign(q_opt[:, 3]).unsqueeze(1))
    nu_opt = -1.*nu_min.unsqueeze(1)
    # Normalize qs (but symeig already does this!)
    # q_opt = qs/torch.norm(q, dim=1).unsqueeze(1) # Unsqueeze as per broadcast rules
    t_solve = time.time() - start
    p = torch.einsum('bn,bnm,bm->b', q_opt, A, q_opt).unsqueeze(1)
    gap = p + nu_opt

    return q_opt, nu_opt, t_solve, gap


def compute_grad_fast(A, nu, q):
    """
    Input: A_vec: (B,4,4) tensor (parametrices B symmetric 4x4 matrices)
           nu: (B,) tensor (optimal lagrange multipliers)
           q: (B,4) tensor (optimal unit quaternions)
    
    Output: grad: (B, 4, 10) tensor (gradient)
           
    Applies the implicit function theorem to compute gradients of qT*A*q s.t |q| = 1, assuming A is symmetric 
    """

    assert(A.dim() > 2 and nu.dim() > 0 and q.dim() > 1)
    

    M = A.new_zeros((A.shape[0], 5, 5))
    I = A.new_zeros((A.shape[0], 4, 4))

    I[:,0,0] = I[:,1,1] = I[:,2,2] = I[:,3,3] = 1.

    M[:, :4, :4] = 2*A + 2.*I*nu.view(-1,1,1)
    M[:, 4,:4] = 2.*q
    M[:, :4,4] = 2.*q

    b = A.new_zeros((A.shape[0], 5, 10))

    #symmetric matrix indices
    idx = torch.triu_indices(4,4)

    i = torch.arange(10)
    I_ij = A.new_zeros((10, 4, 4))

    I_ij[i, idx[0], idx[1]] = 2.
    I_ij[i, idx[1], idx[0]] = 2.
    
    I_ij = I_ij.expand(A.shape[0], 10, 4, 4)

    b[:, :4, :] = torch.einsum('bkij,bi->bjk',I_ij, q) 

    X, _ = torch.solve(b, M)
    grad = -1*X[:,:4,:]
    return grad


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