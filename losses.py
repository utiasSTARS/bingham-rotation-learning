import torch
from quaternions import *
from utils import *

## Quaternions
#Computes q^T A q
def quat_self_supervised_primal_loss(q, A, reduce=True):
    losses = torch.einsum('bn,bnm,bm->b', q, A, q)
    loss = losses.mean() if reduce else losses
    return loss 

def quat_consistency_loss(qs, q_target, reduce=True):
    q = qs[0]
    q_inv = qs[1]
    assert(q.shape == q_inv.shape == q_target.shape)
    d1 = quat_loss(q, q_target, reduce=False)
    d2 = quat_loss(q_inv, quat_inv(q_target), reduce=False)
    d3 = quat_loss(q, quat_inv(q_inv), reduce=False)
    losses =  d1*d1 + d2*d2 + d3*d3
    loss = losses.mean() if reduce else losses
    return loss

def quat_chordal_squared_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses =  2*d*d*(4. - d*d) 
    loss = losses.mean() if reduce else losses
    return loss    

def quat_squared_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses =  0.5*d*d
    loss = losses.mean() if reduce else losses
    return loss

def quat_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses = d
    loss = losses.mean() if reduce else losses
    return loss


## Rotation matrices
def rotmat_frob_squared_norm_loss(C, C_target, reduce=True):
    """Return the Frobenius norm of the difference betwen two batchs of N rotation matrices."""
    assert(C.shape == C_target.shape)
    if C.dim() < 3:
        C = C.unsqueeze(dim=0)
        C_target = C_target.unsqueeze(dim=0)
    losses = (C - C_target).norm(dim=[1,2])**2 #6. - 2.*trace(C.bmm(C_target.transpose(1,2)))
    loss = losses.mean() if reduce else losses
    return loss
