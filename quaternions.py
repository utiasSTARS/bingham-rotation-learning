import torch
import numpy as np
import utils
import math

#NUMPY
##########
def Omega_l(q):
    Om = np.zeros((4,4)) * np.nan
    np.fill_diagonal(Om, q[3]) 
    
    Om[0,1] = -q[2]
    Om[0,2] = q[1]
    Om[0,3] = q[0]

    Om[1,0] = q[2]
    Om[1,2] = -q[0]
    Om[1,3] = q[1]

    Om[2,0] = -q[1]
    Om[2,1] = q[0]
    Om[2,3] = q[2]
    
    Om[3,0] = -q[0]
    Om[3,1] = -q[1]
    Om[3,2] = -q[2]

    return Om

def Omega_r(q):
    Om = np.zeros((4,4)) * np.nan
    np.fill_diagonal(Om, q[3]) 
    
    Om[0,1] = q[2]
    Om[0,2] = -q[1]
    Om[0,3] = q[0]

    Om[1,0] = -q[2]
    Om[1,2] = q[0]
    Om[1,3] = q[1]

    Om[2,0] = q[1]
    Om[2,1] = -q[0]
    Om[2,3] = q[2]
    
    Om[3,0] = -q[0]
    Om[3,1] = -q[1]
    Om[3,2] = -q[2]

    return Om

def pure_quat(v):
    q = np.zeros(4)
    q[:3] = v
    return q

#PYTORCH
##########

#ASSUMES XYZW
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
    assert(q_a.shape == q_b.shape)
    assert(q_a.shape[-1] == 4)
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
        q_b = q_b.unsqueeze(0)
    return torch.min((q_a-q_b).norm(dim=1), (q_a+q_b).norm(dim=1)).squeeze()

def quat_angle_diff(q_a, q_b, units='deg', reduce=True):
    assert(q_a.shape == q_b.shape)
    assert(q_a.shape[-1] == 4)
    diffs = quat_norm_to_angle(quat_norm_diff(q_a, q_b), units=units)
    return diffs.mean() if reduce else diffs

#See Rotation Averaging by Hartley et al. (2013)
def quat_norm_to_angle(q_norms, units='deg'):
    angle = 4.*torch.asin(0.5*q_norms)
    if units == 'deg':
        angle = (180./np.pi)*angle
    elif units == 'rad':
        pass
    else:
        raise RuntimeError('Unknown units in metric conversion.')
    return angle


def quat_to_rotmat(quat, ordering='xyzw'):
    """Form a rotation matrix from a unit length quaternion.

        Valid orderings are 'xyzw' and 'wxyz'.
    """
    if quat.dim() < 2:
        quat = quat.unsqueeze(dim=0)

    if not utils.allclose(quat.norm(p=2, dim=1), 1.):
        print("Warning: Some quaternions not unit length ... normalizing.")
        quat = quat/quat.norm(p=2, dim=1, keepdim=True)

    if ordering is 'xyzw':
        qx = quat[:, 0]
        qy = quat[:, 1]
        qz = quat[:, 2]
        qw = quat[:, 3]
    elif ordering is 'wxyz':
        qw = quat[:, 0]
        qx = quat[:, 1]
        qy = quat[:, 2]
        qz = quat[:, 3]
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

    # Form the matrix
    mat = quat.new_empty(quat.shape[0], 3, 3)

    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    mat[:, 0, 0] = 1. - 2. * (qy2 + qz2)
    mat[:, 0, 1] = 2. * (qx * qy - qw * qz)
    mat[:, 0, 2] = 2. * (qw * qy + qx * qz)

    mat[:, 1, 0] = 2. * (qw * qz + qx * qy)
    mat[:, 1, 1] = 1. - 2. * (qx2 + qz2)
    mat[:, 1, 2] = 2. * (qy * qz - qw * qx)

    mat[:, 2, 0] = 2. * (qx * qz - qw * qy)
    mat[:, 2, 1] = 2. * (qw * qx + qy * qz)
    mat[:, 2, 2] = 1. - 2. * (qx2 + qy2)

    return mat.squeeze_()


#Based on https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
def rotmat_to_quat(mat, ordering='xyzw'):
    """Convert a rotation matrix to a unit length quaternion.

        Valid orderings are 'xyzw' and 'wxyz'.
    """
    if mat.dim() < 3:
        R = mat.unsqueeze(dim=0)
    else:
        R = mat

    assert(R.shape[1] == R.shape[2])
    assert(R.shape[1] == 3)

    #Row first operation
    R = R.transpose(1,2)
    q = R.new_empty((R.shape[0], 4))

    cond1_mask = R[:, 2, 2] < 0.
    cond1a_mask = R[:, 0, 0] > R[:, 1, 1]
    cond1b_mask = R[:, 0, 0] < -R[:, 1, 1]

    if ordering=='xyzw':
        v_ind = torch.arange(0,3)
        w_ind = 3
    else:
        v_ind = torch.arange(1,4)
        w_ind = 0    

    mask = cond1_mask & cond1a_mask
    if mask.any():
        t = 1 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2]
        q[mask, w_ind] =  R[mask, 1, 2]- R[mask, 2, 1]
        q[mask, v_ind[0]] = t
        q[mask, v_ind[1]] = R[mask, 0, 1] + R[mask, 1, 0]
        q[mask, v_ind[2]] = R[mask, 2, 0] + R[mask, 0, 2]
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    mask = cond1_mask & cond1a_mask.logical_not()
    if mask.any():
        t = 1 - R[mask,0, 0] + R[mask,1, 1] - R[mask,2, 2]
        q[mask, w_ind] =  R[mask,2, 0]-R[mask,0, 2]
        q[mask, v_ind[0]] = R[mask,0, 1]+R[mask,1, 0]
        q[mask, v_ind[1]] = t
        q[mask, v_ind[2]] = R[mask,1, 2]+R[mask,2, 1]
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    mask = cond1_mask.logical_not() & cond1b_mask
    if mask.any():
        t = 1 - R[mask,0, 0] - R[mask,1, 1] + R[mask,2, 2]
        q[mask, w_ind] =  R[mask,0, 1]-R[mask,1, 0]
        q[mask, v_ind[0]] = R[mask,2, 0]+R[mask,0, 2]
        q[mask, v_ind[1]] = R[mask,1, 2]+R[mask,2, 1]
        q[mask, v_ind[2]] = t
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    mask = cond1_mask.logical_not() & cond1b_mask.logical_not()
    if mask.any():
        t = 1 + R[mask, 0, 0] + R[mask,1, 1] + R[mask,2, 2]
        q[mask, w_ind] = t
        q[mask, v_ind[0]] = R[mask,1, 2]-R[mask,2, 1]
        q[mask, v_ind[1]] = R[mask,2, 0]-R[mask,0, 2]
        q[mask, v_ind[2]] = R[mask,0, 1]-R[mask,1, 0]
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))
    
    return q.squeeze()


def rotmat_angle_diff(C, C_target, units='deg', reduce=True):
    assert(C.shape == C_target.shape)
    if C.dim() < 3:
        C = C.unsqueeze(dim=0)
        C_target = C_target.unsqueeze(dim=0)

    rotmat_frob_norms = (C - C_target).norm(dim=[1,2]) #torch.sqrt(6. - 2.*trace(C.bmm(C_target.transpose(1,2))))
    diffs = rotmat_frob_norm_to_angle(rotmat_frob_norms, units=units)
    return diffs.mean() if reduce else diffs

#See Rotation Averaging by Hartley et al. (2013)
def rotmat_frob_norm_to_angle(frob_norms, units='deg'):
    sin = torch.clamp(0.25*math.sqrt(2)*frob_norms, min=-1., max=1.)
    angle = 2.*torch.asin(sin)
    if units == 'deg':
        angle = (180./np.pi)*angle
    elif units == 'rad':
        pass
    else:
        raise RuntimeError('Unknown units in metric conversion.')
    return angle
