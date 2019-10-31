import torch
from torch.autograd import gradcheck
import numpy as np
from convex_wahba import solve_wahba, compute_grad, gen_sim_data, build_A

class SimpleNetwork(torch.nn.Module):
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
        y = self.net(x)
        return y

class QuadQuat(torch.autograd.Function):
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
        q_opt = torch.from_numpy(q_opt).double()
        nu_opt = torch.tensor(nu_opt, dtype=torch.double)
        ctx.save_for_backward(A, q_opt, nu_opt)

        return q_opt

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        A, q_opt, nu_opt = ctx.saved_tensors
        grad_qcqp = torch.from_numpy(
            compute_grad(A.detach().numpy(), nu_opt.item(), q_opt.detach().numpy())
            )
        outgrad = torch.einsum('bij,b->ij', grad_qcqp, grad_output) 
        return outgrad

def check_gradient():
    qcqp_solver = QuadQuat.apply
    iters = 20
    #N, sigma = 100, 0.01
    #_, x_1, x_2 = gen_sim_data(N, sigma)
    # A = build_A(x_1, x_2, sigma*sigma*np.ones(N))
    # A = torch.from_numpy(A)
    # A.requires_grad_(True)

    for i in range(iters):
        A = torch.randn((4,4), dtype=torch.double, requires_grad=True)
        input = (A,)
        grad_test = gradcheck(qcqp_solver, input, eps=1e-6, atol=1e-4)
        print('Grad check iteration {}/{}...Passed.'.format(i, iters))

    
    print('Passed all gradient checks!')


if __name__=='__main__':
    check_gradient()