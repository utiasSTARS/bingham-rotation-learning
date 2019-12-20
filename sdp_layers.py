import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from rotation_matrix_sdp import rotation_matrix_constraints
from convex_layers import *
import time
from utils import allclose
from quaternions import *

def x_from_xxT(xxT):
    """
    Input: BxNxN symmetric rank 1 tensor 
    Output: BxN tensor x s.t. xxT = x*x.T (outer product), assumes last element must be positive to resolve sign ambiguities
    """

    if xxT.dim() < 3:
        xxT = xxT.unsqueeze(dim=0)
    assert(xxT.shape[1] == xxT.shape[2])
    N = xxT.shape[1]
    x = torch.sqrt(xxT[:, torch.arange(N), torch.arange(N)])
    signs = torch.sign(xxT[:, :, -1])
    x = x * signs
    return x.squeeze()


class RotMatSDPSolver(torch.nn.Module):
    def __init__(self):
        super(RotMatSDPSolver, self).__init__()
        
        X = cp.Variable((10, 10), PSD=True)
        constraint_matrices, c_vec = rotation_matrix_constraints()
        constraints = [cp.trace(constraint_matrices[idx, :, :] @ X) == c_vec[idx]
                    for idx in range(constraint_matrices.shape[0])]
        A = cp.Parameter((10, 10), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.trace(A @ X)), constraints)
        self.sdp_solver = CvxpyLayer(prob, parameters=[A], variables=[X])
    
    def forward(self, A_vec):
        
        if A_vec.dim() < 2:
            A_vec = A_vec.unsqueeze(dim=0)

        A = convert_Avec_to_A(A_vec)
        X, = self.sdp_solver(A)

        x = x_from_xxT(X)

        if x.dim() < 2:
            x = x.unsqueeze(dim=0)

        r_vec = x[:, :9]
        rotmat = r_vec.view(-1, 3,3).transpose(1,2)
        return rotmat.squeeze()


def compare_solver_time():
    num_samples = 1000
    sdp_rot_solver = RotMatSDPSolver()
    A_vec = torch.randn((num_samples, 55), dtype=torch.double, requires_grad=True)
    start = time.time()
    rotmat = sdp_rot_solver(A_vec)
    rotmat.sum().backward()
    print('Solved {} SDPs in {:.3F} sec using cvxpylayers.'.format(num_samples, time.time() - start))


    start = time.time()
    r = HomogeneousRotationQCQPFastSolver.apply(A_vec)
    rotmat_custom = r[:, :9].view(-1,3,3).transpose(1,2)
    print('Solved {} SDPs in {:.3F} sec using custom solver.'.format(num_samples, time.time() - start))
    print('Mean angle diff : {:.3E} deg'.format(rotmat_angle_diff(rotmat, rotmat_custom)))


    qcqp_solver = QuadQuatFastSolver.apply
    A_vec = torch.randn((num_samples, 10), dtype=torch.double, requires_grad=True)

    start = time.time()
    q = qcqp_solver(A_vec)
    print('Solved {} Quat QCQPs in {:.3F} sec.'.format(num_samples, time.time() - start))

if __name__ == '__main__':

    num_samples = 1000
    sdp_rot_solver = RotMatSDPSolver()
    A_vec = torch.randn((num_samples, 55), dtype=torch.double, requires_grad=True)
    A_vec = convert_Avec_to_Avec_psd(A_vec)

    start = time.time()
    rotmat = sdp_rot_solver(A_vec)
    print('Solved {} SDPs in {:.3F} sec using cvxpylayers.'.format(num_samples, time.time() - start))

    rotmat.sum().backward()

