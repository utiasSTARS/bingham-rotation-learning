import torch
from torch.autograd import gradcheck
import numpy as np
from nets_and_solvers import QuadQuatSolver, QuadQuatFastSolver
from convex_wahba import solve_wahba, solve_wahba_fast, compute_grad, build_A
from helpers import matrix_diff, so3_diff, gen_sim_data, solve_horn
from liegroups.numpy import SO3

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def test_pytorch_analytic_gradient(eps=1e-6, tol=1e-4, num_samples=3):
    print('Checking PyTorch gradients (random A, batch_size: {})'.format(num_samples))
    qcqp_solver = QuadQuatSolver.apply
    A = torch.randn((num_samples,4,4), dtype=torch.double, requires_grad=True)
    A = 0.5 * (A.transpose(1, 2) + A)
    input = (A,)
    grad_test = gradcheck(qcqp_solver, input, eps=eps, atol=tol)
    assert(grad_test == True)
    print('Batch...Passed.')


def test_pytorch_fast_analytic_gradient(eps=1e-6, tol=1e-4, num_samples=10):
    print('Checking PyTorch sped-up gradients (random A, batch_size: {})'.format(num_samples))
    qcqp_solver = QuadQuatFastSolver.apply
    A_vec = torch.randn((num_samples, 10), dtype=torch.double, requires_grad=True)
    input = (A_vec,)
    grad_test = gradcheck(qcqp_solver, input, eps=eps, atol=tol)
    assert (grad_test == True)
    print('Batch...Passed.')


def test_duality_gap_wahba_solver(num_samples=100):
    print('Checking duality gap on the fast Wahba solver')
    A = torch.randn((num_samples, 4, 4), dtype=torch.double, requires_grad=True)
    A = 0.5 * (A.transpose(1, 2) + A)
    _, _, _, gap = solve_wahba_fast(A)
    assert np.allclose(gap.detach().numpy(), 0.0)
    print('Done')


def test_compare_fast_and_slow_solvers(eps=1e-6, tol=1e-4, num_samples=5):
    print('Checking accuracy of fast solver')
    # qcqp_solver = QuadQuatSolver.apply
    # qcqp_solver_fast = QuadQuatFastSolver.apply
    A = torch.randn((num_samples, 4, 4), dtype=torch.double, requires_grad=True)
    A = 0.5*(A.transpose(1, 2) + A)
    # input = (A,)
    q_out = QuadQuatSolver.apply(A).detach().numpy()
    q_out_fast = QuadQuatFastSolver.apply(A).detach().numpy()
    q_out_diff = np.minimum(np.abs(q_out_fast-q_out), np.abs(q_out_fast+q_out))
    print(np.max(q_out_diff))
    assert np.allclose(q_out_diff, 0., atol=1e-5)
    # print(q_out)
    # print(q_out_fast)

def test_pytorch_manual_analytic_gradient(eps=1e-6, tol=1e-4, num_samples=1):
    A = torch.randn((num_samples, 4, 4), dtype=torch.double, requires_grad=False)
    A = 0.5*(A.transpose(1, 2) + A)
    q_opt, nu_opt, _, _ = solve_wahba_fast(A, redundant_constraints=True)

    for i in range(num_samples):
        G_numerical = numerical_grad_fast(A[i], eps)
        G_analytic = torch.from_numpy(compute_grad(A[i].detach().numpy(), nu_opt[i].detach().numpy(), q_opt[i].detach().numpy()))
        
        rel_diff = matrix_diff(G_analytic.numpy(), G_numerical.numpy())

        print(G_numerical - G_analytic)
        assert(rel_diff < tol)
        print('Sample {}/{}...Passed.'.format(i+1, num_samples))

def numerical_grad_fast(A, eps):
    if A.dim() < 3:
        A = A.unsqueeze(0)

    G_numerical = torch.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            dA_ij = torch.zeros((1,4,4))
            dA_ij[0,i,j] = eps
            dA_ij[0,j,i] = eps
            q_plus,_,_,_ =  solve_wahba_fast(A+dA_ij, redundant_constraints=True)
            q_minus,_,_,_ =  solve_wahba_fast(A-dA_ij, redundant_constraints=True)
            G_numerical[:,i,j] = (q_plus.squeeze() - q_minus.squeeze())/(2.*eps)
    return G_numerical

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
    print('Checking NumPy gradients (random A)...')
    A = np.random.randn(4,4)
    q_opt, nu_opt, _, _ = solve_wahba(A, redundant_constraints=True)

    for i in range(num_samples):
        G_numerical = numerical_grad(A, eps)
        G_analytic = compute_grad(A, nu_opt, q_opt)
        rel_diff = matrix_diff(G_analytic, G_numerical)
        assert(rel_diff < tol)
        print('Sample {}/{}...Passed.'.format(i+1, num_samples))

def test_numpy_solver(N=100, sigma=0.01, tol=1.):
    #Tolerance in terms of degrees

    C, x_1, x_2 = gen_sim_data(N, sigma)
    redundant_constraints = True


    ## Solver
    print('Checking single solve with synthetic data...')
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
    # test_numpy_solver()
    # print("=============")
    # test_numpy_analytic_gradient()
    # print("=============")
    # test_pytorch_analytic_gradient()
    # print("=============")
    # test_pytorch_fast_analytic_Gradient()
    # # print("=============")
    # test_compare_fast_and_slow_solvers()
    # # print("=============")
    # test_duality_gap_wahba_Solver()

    test_pytorch_fast_analytic_gradient()
    #test_pytorch_manual_analytic_gradient()
