import torch
from torch.autograd import gradcheck
import numpy as np
from nets_and_solvers import QuadQuatSolver
from convex_wahba import solve_wahba, compute_grad, build_A
from helpers import matrix_diff, so3_diff, gen_sim_data, solve_horn
from liegroups.numpy import SO3

def test_pytorch_analytic_gradient(eps=1e-6, tol=1e-4, num_samples=5):
    print('Checking PyTorch gradients...')
    qcqp_solver = QuadQuatSolver.apply
    for i in range(num_samples):
        A = torch.randn((4,4), dtype=torch.double, requires_grad=True)
        input = (A,)
        grad_test = gradcheck(qcqp_solver, input, eps=eps, atol=tol)
        assert(grad_test == True)
        print('PyTorch gradcheck A sample {}/{}...Passed.'.format(i+1, num_samples))


def numerical_grad(A, eps):
    G_numerical = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            dA_ij = np.zeros((4,4))
            dA_ij[i,j] = eps
            q_plus,_,_,_ =  solve_wahba(A+dA_ij, redundant_constraints=True)
            q_minus,_,_,_ =  solve_wahba(A-dA_ij, redundant_constraints=True)
            G_numerical[:,i,j] = (q_plus - q_minus)/(2.*eps)
    return G_numerical

def test_numpy_analytic_gradient(eps=1e-6, tol=1e-4, num_samples=5):
    print('Checking NumPy gradients...')
    A = np.random.randn(4,4)
    q_opt, nu_opt, _, _ = solve_wahba(A, redundant_constraints=True)

    for i in range(num_samples):
        G_numerical = numerical_grad(A, eps)
        G_analytic = compute_grad(A, nu_opt, q_opt)
        rel_diff = matrix_diff(G_analytic, G_numerical)
        assert(rel_diff < tol)
        print('NumPy gradcheck A sample {}/{}...Passed.'.format(i+1, num_samples))

def test_numpy_solver(N=500, sigma=0.01, tol=1.):
    #Tolerance in terms of degrees
    
    C, x_1, x_2 = gen_sim_data(N, sigma)
    redundant_constraints = True

    ## Solver
    print('Checking single solve...')
    A = build_A(x_1, x_2, sigma*sigma*np.ones(N))
    #A = np.random.randn(4,4)
    q_opt, _, t_solve, gap = solve_wahba(A, redundant_constraints=redundant_constraints)
    C_opt = SO3.from_quaternion(q_opt, ordering='xyzw').as_matrix()

    ## Output
    print('Done. Solved in {:.3f} seconds.'.format(t_solve))
    print('Duality gap: {:.3E}.'.format(gap))
    #Compare to known rotation
    print('Convex rotation error: {:.3f} deg'.format(so3_diff(C_opt, C)))
    print('Horn rotation error: {:.3f} deg'.format(so3_diff(solve_horn(x_1, x_2), C)))

    assert(so3_diff(C_opt, C) < tol)

if __name__=='__main__':
    test_numpy_solver()
    print("=============")
    test_numpy_analytic_gradient()
    print("=============")
    test_pytorch_analytic_gradient()
