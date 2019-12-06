import torch
import numpy as np

def quat_inv(q):
    #Note, 'empty_like' is necessary to prevent in-place modification (which is not auto-diff'able)
    if q.dim() < 2:
        q = q.unsqueeze()
    q_inv = torch.empty_like(q)
    q_inv[:, :3] = -1*q[:, :3]
    q_inv[:, 3] = q[:, 3]
    return q_inv.squeeze()
    
#Quaternion difference of two unit quaternions
def quat_norm_diff(q_a, q_b):
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
    if q_b.dim() < 2:
        q_b = q_b.unsqueeze(0)
    return torch.min((q_a-q_b).norm(dim=1), (q_a+q_b).norm(dim=1)).squeeze_()

def quat_angle_diff(q, q_target, units='deg', reduce=True):
    assert(q.shape == q_target.shape)
    diffs = quat_norm_to_angle(quat_norm_diff(q, q_target), units=units)
    return diffs.mean() if reduce else diffs

#See Rotation Averaging by Hartley et al. (2013)
def quat_norm_to_angle(q_met, units='deg'):
    angle = 4.*torch.asin(0.5*q_met)
    if units == 'deg':
        angle = (180./np.pi)*angle
    elif units == 'rad':
        pass
    else:
        raise RuntimeError('Unknown units in metric conversion.')
    return angle

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

