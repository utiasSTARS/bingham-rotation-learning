
import torch
from torch.autograd import gradcheck
import numpy as np
from convex_wahba import solve_wahba, solve_wahba_fast, compute_grad, gen_sim_data, build_A, compute_grad_fast
import torch.nn.functional as F

#Utility module to replace BatchNorms without affecting structure
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



class QuatNet(torch.nn.Module):
    def __init__(self, A_net=None):
        super(QuatNet, self).__init__()
        if A_net is None:
            raise RuntimeError('Must pass in an ANet to QuatNet')
        self.A_net = A_net
        self.qcqp_solver = QuadQuatFastSolver.apply

    def forward(self, x, A_prior=None):
        A_vec = self.A_net(x, A_prior)
        if self.A_net.bidirectional:
            q = self.qcqp_solver(A_vec[0])
            q_inv = self.qcqp_solver(A_vec[1])
            return [q, q_inv]
        else:
            q = self.qcqp_solver(A_vec)
            return q


class APriorNet(torch.nn.Module):
    def __init__(self):
        super(APriorNet, self).__init__()
        self.fc1 = torch.nn.Linear(10,10)
        self.bn1 = torch.nn.BatchNorm1d(10)

    def forward(self, A_vec):
        A_vec = F.relu(self.bn1(self.fc1(A_vec))) + A_vec
        return A_vec

class ANet(torch.nn.Module):
    def __init__(self, num_pts, bidirectional=True):
        super(ANet, self).__init__()
        self.num_pts = num_pts
        self.bidirectional = bidirectional #Evaluate both forward and backward directions
        self.A_prior_net = APriorNet()
        self.feat_net1 = FCPointFeatNet(num_pts=num_pts)
        self.feat_net2 = FCPointFeatNet(num_pts=num_pts)
        
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc_out = torch.nn.Linear(256, 10)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.activ1 = torch.nn.ELU()
        self.activ2 = torch.nn.ELU()


    def feats_to_A(self, x):
        x = self.activ1(self.bn1(self.fc1(x)))
        x = self.activ2(self.bn2(self.fc2(x)))
        A_vec = self.fc_out(x)
        return A_vec

    def forward(self, x, A_prior=None):
        #Decompose input into two point clouds
        x_1, x_2 = torch.chunk(x, 2, dim=1)
        #x_1, x_2 are Bx3xN where B is the minibatch size, N is num_pts
        x_1 = x_1.view(-1, self.num_pts, 3).transpose(1,2)
        x_2 = x_2.view(-1, self.num_pts, 3).transpose(1,2)
        
        #Collect and concatenate features
        #x_1 -> x_2
        feats_12 = torch.cat([self.feat_net1(x_1), self.feat_net2(x_1)], dim=1)
        A1 = self.feats_to_A(feats_12)
        
        #Prior? Doesn't make sense with symmetric loss unless we give two priors...TODO
        if A_prior is not None:
            A1 = A1 + self.A_prior_net(A_prior)

        if self.bidirectional:
            #x_2 -> x_1
            feats_21 = torch.cat([self.feat_net1(x_2), self.feat_net2(x_1)], dim=1)
            A2 = self.feats_to_A(feats_21)
            return [A1, A2]

        return A1


class QuadQuatFastSolver(torch.autograd.Function):
    """
    TODO: - pytorch tutorial,
          - fast eigenvalue solution forward solve,
          - remove numpy dependencies in backwards pass
    Differentiable QCQP solver
    Input: Bx10 tensor 'A_vec' which encodes symmetric 4x4 matrices, A
    Output: q that minimizes q^T A q s.t. |q| = 1
    """


    @staticmethod
    def forward(ctx, A_vec):

        if A_vec.dim() < 2:
            A_vec = A_vec.unsqueeze()

        #Convert Bx10 tensor to Bx4x4 symmetric matrices
        idx = torch.triu_indices(4,4)
        A = A_vec.new_zeros((A_vec.shape[0],4,4))   
        A[:, idx[0], idx[1]] = A_vec
        A[:, idx[1], idx[0]] = A_vec

        q, nu, _, _ = solve_wahba_fast(A)
        ctx.save_for_backward(A, q, nu)

        return q

    @staticmethod
    def backward(ctx, grad_output):
        A, q, nu = ctx.saved_tensors
        grad_qcqp = compute_grad_fast(A, nu, q)
        outgrad = torch.einsum('bkq,bk->bq', grad_qcqp, grad_output)
        return outgrad

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
        # print('Slow method q and nu')
        # print(q)
        # print(nu)
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

#Modelled after PointNet, see https://github.com/kentsyx/pointnet-pytorch/blob/master/pointnet.py
#Outputs a 3x3 affine transformation
class AffineNet(torch.nn.Module):
    def __init__(self):
        super(AffineNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 9)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(-1,3,3) 
        #I = torch.zeros_like(x)
        #I[:,0,0] = I[:,1,1] = I[:,2,2] = 1.
        #x += I
        return x

class FCPointFeatNet(torch.nn.Module):
    def __init__(self, num_pts):
        super(FCPointFeatNet, self).__init__()
        self.num_pts = num_pts
        self.cnn_in = torch.nn.Conv2d(3, 128, 1, 1, 0)
        self.cnn = torch.nn.Conv2d(128, 128, 1, 1, 0)
        self.fc_out = torch.nn.Linear(128*num_pts, 512)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.bn2 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        x = x.unsqueeze(3)
        x = F.relu(F.instance_norm(self.bn1(self.cnn_in(x))))
        x = F.relu(F.instance_norm(self.bn2(self.cnn(x)))) + x
        x = self.fc_out(x.view(-1, 128*self.num_pts))
        return x

class PointFeatNet(torch.nn.Module):
    def __init__(self):
        super(PointFeatNet, self).__init__()
        self.affine_net = AffineNet()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(512)

    def forward(self, x):
        M = self.affine_net(x)
        #Apply affine transform
        x = M.bmm(x)   
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        return x
