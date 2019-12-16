import numpy as np
import scipy as sp
from liegroups.numpy import SO3
import cvxpy as cp
import time
import torch  

def normalize_Avec(A_vec):
    """ Normalizes BxM vectors such that resulting symmetric BxNxN matrices have unit Frobenius norm"""
    """ M = N*(N+1)/2"""
    
    A = convert_Avec_to_A(A_vec, A_dim)
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
        raise ValueError("Arbitrary A_vec not yet implementedf")
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

#=========================PYTORCH (FAST) SOLVER=========================

class QuadQuatFastSolver(torch.autograd.Function):
    """
    Differentiable QCQP solver
    Input: Bx10 tensor 'A_vec' which encodes symmetric 4x4 matrices, A
    Output: q that minimizes q^T A q s.t. |q| = 1
    """

    @staticmethod
    def forward(ctx, A_vec):

        if A_vec.dim() < 2:
            A_vec = A_vec.unsqueeze()

        A = convert_Avec_to_A(A_vec)
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
    # Normalize qs (but symeig already does this!)
    # q_opt = qs/torch.norm(q, dim=1).unsqueeze(1) # Unsqueeze as per broadcast rules
    #t_solve = time.time() - start
    if compute_gap:
        p = torch.einsum('bn,bnm,bm->b', q_opt, A, q_opt).unsqueeze(1)
        gap = p + nu_opt
        return q_opt, nu_opt, gap
    #t_solve = 0
    #gap = 10

    return q_opt, nu_opt#, t_solve, gap


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



#=========================NUMPY/CVXPY (SLOW) SOLVER=========================

class QuadQuatSolver(torch.autograd.Function):
    """
    Differentiable QCQP solver
    Input: 4x4 matrix A
    Output: q that minimizes q^T A q s.t. |q| = 1
    """

    @staticmethod
    def forward(ctx, A):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        if A.dim() > 2:
            # minibatch size > 1
            # Iterate for now, maybe speed this up later
            q = torch.empty(A.shape[0], 4, dtype=torch.double)
            nu = torch.empty(A.shape[0], 1, dtype=torch.double)
            for i in range(A.shape[0]):
                try:
                    q_opt, nu_opt, _, _ = solve_wahba(A[i].detach().numpy(),redundant_constraints=True)
                    q[i] = torch.from_numpy(q_opt)
                    nu[i,0] = nu_opt
                except:
                    raise RuntimeError('Wahba Solve failed!')
                    
        else:
            q_opt, nu_opt, _, _ = solve_wahba(A.detach().numpy(),redundant_constraints=True)
            q = torch.from_numpy(q_opt)
            nu = nu_opt*torch.ones(1, dtype=torch.double)

        ctx.save_for_backward(A, q, nu)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        A, q, nu = ctx.saved_tensors

        if A.dim() > 2:
            # minibatch size > 1
            # Iterate for now, maybe speed this up later
            grad_qcqp = torch.empty(A.shape[0], 4, 4, 4, dtype=torch.double)
            for i in range(A.shape[0]):
                grad_qcqp[i] = torch.from_numpy(compute_grad(A[i].detach().numpy(), nu[i].detach().numpy()[0], q[i].detach().numpy())).double()
            outgrad = torch.einsum('bkij,bk->bij', grad_qcqp, grad_output) 
        else:
            grad_qcqp = torch.from_numpy(compute_grad(A.detach().numpy(), nu.detach().numpy()[0], q.detach().numpy())).double()
            outgrad = torch.einsum('kij,k->ij', grad_qcqp, grad_output) 

        return outgrad
        
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

def q_from_qqT(qqT):
    #Returns unit quaternion q from q * q^T 4x4 matrix
    #Assumes scalar is the last value and it is positive (can make this choice since q = -q)

    q = np.sqrt(np.diag(qqT))
    if qqT[0,3] < 0.:
        q[0] *=  -1.
    if qqT[1,3] < 0.:
        q[1] *=  -1.
    if qqT[2,3] < 0.:
        q[2] *=  -1.

    return q