import numpy as np
import scipy as sp
import time
import torch

def normalize_Avec(A_vec):
    """ Normalizes BxM vectors such that resulting symmetric BxNxN matrices have unit Frobenius norm"""
    """ M = N*(N+1)/2"""
    
    A = convert_Avec_to_A(A_vec)
    if A.dim() < 3:
        A = A.unsqueeze(dim=0)
    A = A / A.norm(dim=[1,2], keepdim=True)
    return convert_A_to_Avec(A).squeeze()

def convert_A_to_Avec(A):
    """ Convert BxNXN symmetric matrices to BxM vectors encoding unique values"""
    if A.dim() < 3:
        A = A.unsqueeze(dim=0)
    idx = torch.triu_indices(A.shape[1], A.shape[1])
    A_vec = A[:, idx[0], idx[1]]
    return A_vec.squeeze()

def convert_Avec_to_A(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    """ M = N*(N+1)/2"""
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)
    
    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 55:
        A_dim = 10
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.triu_indices(A_dim,A_dim)
    A = A_vec.new_zeros((A_vec.shape[0],A_dim,A_dim))   
    A[:, idx[0], idx[1]] = A_vec
    A[:, idx[1], idx[0]] = A_vec
    return A.squeeze()

def convert_Avec_to_Avec_psd(A_vec):
    """ Convert BxM tensor (encodes symmetric NxN amatrices) to BxM tensor  
    (encodes symmetric and PSD 4x4 matrices)"""

    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze()
    
    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 55:
        A_dim = 10
    else:
        raise ValueError("Arbitrary A_vec not yet implementedf")

    idx = torch.tril_indices(A_dim,A_dim)
    L = A_vec.new_zeros((A_vec.shape[0],A_dim,A_dim))   
    L[:, idx[0], idx[1]] = A_vec
    A = L.bmm(L.transpose(1,2))
    A_vec_psd = convert_A_to_Avec(A)
    return A_vec_psd



def A_vec_to_quat(A_vec):
    A = convert_Avec_to_A(A_vec)
    if A.dim() < 3:
        A = A.unsqueeze(dim=0)
    _, evs = torch.symeig(A, eigenvectors=True)
    return evs[:,:,0].squeeze()


# #=========================PYTORCH (FAST) SOLVER=========================

class QuadQuatFastSolver(torch.autograd.Function):
    """
    Differentiable QCQP solver
    Input: Bx10 tensor 'A_vec' which encodes symmetric 4x4 matrices, A
    Output: q that minimizes q^T A q s.t. |q| = 1
    """

    @staticmethod
    def forward(ctx, A_vec):

        A = convert_Avec_to_A(A_vec)
        if A.dim() < 3:
            A = A.unsqueeze(dim=0)
        q, nu  = solve_wahba_fast(A)
        ctx.save_for_backward(A, q, nu)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        A, q, nu = ctx.saved_tensors
        grad_qcqp = compute_grad_fast(A, nu, q)
        outgrad = torch.einsum('bkq,bk->bq', grad_qcqp, grad_output)
        return outgrad

def solve_wahba_fast(A, compute_gap=False):
    """
    Use a fast eigenvalue solution to the dual of the 'generalized Wahba' problem to solve the primal.
    :param A: quadratic cost matrix
    :param redundant_constraints: boolean indicating whether to use redundand constraints
    :return: Optimal q, optimal dual var. nu, time to solve, duality gap
    """
    #start = time.time()
    # Returns (b,n) and (b,n,n) tensors
    nus, qs = torch.symeig(A, eigenvectors=True)
    nu_min, nu_argmin = torch.min(nus, 1)# , keepdim=False, out=None)
    q_opt = qs[torch.arange(A.shape[0]), :, nu_argmin]
    q_opt = q_opt*(torch.sign(q_opt[:, 3]).unsqueeze(1))
    nu_opt = -1.*nu_min.unsqueeze(1)
    if compute_gap:
        p = torch.einsum('bn,bnm,bm->b', q_opt, A, q_opt).unsqueeze(1)
        gap = p + nu_opt
        return q_opt, nu_opt, gap
    return q_opt, nu_opt

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

    M[:, :4, :4] = A + I*nu.view(-1,1,1)
    M[:, 4,:4] = q
    M[:, :4,4] = q

    b = A.new_zeros((A.shape[0], 5, 10))

    #symmetric matrix indices
    idx = torch.triu_indices(4,4)

    i = torch.arange(10)
    I_ij = A.new_zeros((10, 4, 4))

    I_ij[i, idx[0], idx[1]] = 1.
    I_ij[i, idx[1], idx[0]] = 1.
    
    I_ij = I_ij.expand(A.shape[0], 10, 4, 4)

    b[:, :4, :] = torch.einsum('bkij,bi->bjk',I_ij, q) 

    #This solves all gradients simultaneously!
    X, _ = torch.solve(b, M)
    grad = -1*X[:,:4,:]
    return grad