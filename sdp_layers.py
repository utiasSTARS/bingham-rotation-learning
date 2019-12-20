import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from rotation_matrix_sdp import rotation_matrix_constraints


def make_rotation_matrix_sdp_layer():
    X = cp.Variable((10, 10), PSD=True)
    constraint_matrices, c_vec = rotation_matrix_constraints()
    constraints = [cp.trace(constraint_matrices[idx, :, :] @ X) == c_vec[idx]
                   for idx in range(constraint_matrices.shape[0])]
    A = cp.Parameter((10, 10), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.trace(A @ X)), constraints)
    return CvxpyLayer(prob, parameters=[A], variables=[X])

if __name__ == '__main__':

    # Sample code
    # n, m = 2, 3
    # x = cp.Variable(n)
    # A = cp.Parameter((m, n))
    # b = cp.Parameter(m)
    # constraints = [x >= 0]
    # objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    # problem = cp.Problem(objective, constraints)
    # assert problem.is_dpp()
    #
    # cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    # A_tch = torch.randn(m, n, requires_grad=True)
    # b_tch = torch.randn(m, requires_grad=True)
    #
    #
    # # solve the problem
    # solution, = cvxpylayer(A_tch, b_tch)
    #
    # # compute the gradient of the sum of the solution with respect to A, b
    # solution.sum().backward()

    sdp_rot_layer = make_rotation_matrix_sdp_layer()
