
import torch
from torch.autograd import gradcheck
import numpy as np
from convex_wahba import solve_wahba, compute_grad, gen_sim_data, build_A

class ANetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SimpleNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        A = self.net(x).view(-1, 4, 4)
        return A

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
        grad_qcqp = torch.from_numpy(
            compute_grad(A.detach().numpy(), nu.detach().numpy()[0], q.detach().numpy())
            ).double()
        outgrad = torch.einsum('bij,b->ij', grad_qcqp, grad_output) 

        return outgrad
