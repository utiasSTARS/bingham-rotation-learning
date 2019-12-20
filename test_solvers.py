import torch
from torch.autograd import gradcheck
import numpy as np
from liegroups.numpy import SO3
from convex_layers import *
from quaternions import *
from helpers_sim import *
import os
from sdp_layers import RotMatSDPSolver

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

def test_rotmat_pytorch_analytic_gradient(eps=1e-6, tol=1e-4, num_samples=2):
    print('Checking PyTorch rotmat gradients (random A, batch_size: {})'.format(num_samples))
    qcqp_solver = HomogeneousRotationQCQPFastSolver.apply
    A_vec = torch.randn((num_samples, 55), dtype=torch.double, requires_grad=True)
    #A_vec = convert_Avec_to_Avec_psd(A_vec)
    input = (A_vec,)
    grad_test = gradcheck(qcqp_solver, input, eps=eps, atol=tol)
    assert (grad_test == True)
    print('Batch...Passed.')

def test_rotmat_sdp_pytorch_analytic_gradient(eps=1e-6, tol=1e-4, num_samples=2):
    print('Checking PyTorch / CVXPYLayers SDP solver gradients (random A, batch_size: {})'.format(num_samples))
    sdp_solver = RotMatSDPSolver()
    A_vec = torch.randn((num_samples, 55), dtype=torch.double, requires_grad=True)
    A_vec = convert_Avec_to_Avec_psd(A_vec)
    A_vec = normalize_Avec(A_vec)

    # A, _ = create_wahba_As(num_samples)
    # A_vec = convert_A_to_Avec(A)
    # A_vec.requires_grad = True

    input = (A_vec,)
    grad_test = gradcheck(sdp_solver, input, eps=eps, atol=tol)
    assert (grad_test == True)
    print('Batch...Passed.')


def test_pytorch_fast_analytic_gradient(eps=1e-6, tol=1e-4, num_samples=100):
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
    _, _, gap = solve_wahba_fast(A, compute_gap=True)
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
    q_out_fast = QuadQuatFastSolver.apply(convert_A_to_Avec(A)).detach().numpy()
    q_out_diff = np.minimum(np.abs(q_out_fast-q_out), np.abs(q_out_fast+q_out))
    print(np.max(q_out_diff))
    assert np.allclose(q_out_diff, 0., atol=1e-5)
    # print(q_out)
    # print(q_out_fast)

def test_rotmat_wahba():
    print('Checking accuracy of QCQP rotmat solver')
    N = 1000
    sigma = 0.
    C = SO3.exp(np.random.randn(3)).as_matrix()
    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = normalized(np.random.randn(N, 3), axis=1)
    #Rotate and add noise
    noise = np.random.randn(N,3)
    noise = (noise.T*sigma).T
    x_2 = C.dot(x_1.T).T + noise

    A = np.zeros((10,10))
    for i in range(N):
        mat = np.zeros((3,10))
        mat[:,:9] = np.kron(x_1[i], np.eye(3))
        mat[:,9] = -x_2[i]
        A += mat.T.dot(mat)


    constraint_matrices, c_vec = rotation_matrix_constraints()
    _, C_solve = solve_equality_QCQP_dual(A, constraint_matrices, c_vec)

    print('Ground truth:')
    print(C)
    print('Solved:')
    print(C_solve)
    print('Angle difference: {:.3E} deg'.format(rotmat_angle_diff(torch.from_numpy(C), torch.from_numpy(C_solve), units='deg').item()))


#Creates num 55x55 A matrices and associated rotation matrices C
#Based on the point-to-point rotation matrix wahba formulation
def create_wahba_As(N=10):
    A = torch.zeros(N, 10, 10, dtype=torch.double)
    C = torch.empty(N ,3, 3, dtype=torch.double)
    N_points = 100
    for n in range(N):
        sigma = 0.
        C_n = SO3.exp(np.random.randn(3)).as_matrix()
        C[n] = torch.from_numpy(C_n)
        #Create two sets of vectors (normalized to unit l2 norm)
        x_1 = normalized(np.random.randn(N_points, 3), axis=1)
        #Rotate and add noise
        noise = np.random.randn(N_points,3)
        noise = (noise.T*sigma).T
        x_2 = C_n.dot(x_1.T).T + noise

        for i in range(N_points):
            mat = np.zeros((3,10))
            mat[:,:9] = np.kron(x_1[i], np.eye(3))
            mat[:,9] = -x_2[i]
            A[n] += torch.from_numpy(mat.T.dot(mat))
    return A, C

def test_rotmat_sdp_wahba():
    N = 1000
    print('Checking accuracy of SDP rotmat solver with {} datasets.'.format(N))
    A, C = create_wahba_As(N)
    sdp_solver = RotMatSDPSolver()
    A_vec = convert_A_to_Avec(A)

    start = time.time()
    C_solve = sdp_solver(A_vec)
    mean_error = rotmat_angle_diff(C, C_solve, units='deg').item()

    print('Mean angle error: {:.3E} deg. Total solve time: {:.3F} sec.'.format(mean_error, time.time() - start))

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
    # test_pytorch_fast_analytic_gradient()
    # print("=============")
    # test_compare_fast_and_slow_solvers()
    # print("=============")
    # test_duality_gap_wahba_solver()

    # print("=============")
    # test_pytorch_manual_analytic_gradient()

    print('===============')
    test_rotmat_sdp_wahba()
    #test_rotmat_wahba()
    # print("=============")
    # test_rotmat_pytorch_analytic_gradient()
    #print("=============")
    #test_rotmat_sdp_pytorch_analytic_gradient()