import torch
import numpy as np
import utils

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


#WARNING NOTE: THIS IS WXYZ
#TODO: ADD OPTION FOR XYZW 
#=======================
def quat_exp(phi):
    # input: phi: Nx3
    # output: Exp(phi) Nx4 (see Sola eq. 101)

    if phi.dim() < 2:
        phi = phi.unsqueeze(0)

    q = phi.new_empty((phi.shape[0], 4))
    phi_norm = phi.norm(dim=1, keepdim=True)
    q[:,0] = torch.cos(phi_norm.squeeze()/2.)
    q[:, 1:] = (phi/phi_norm)*torch.sin(phi_norm/2.)
    return q.squeeze(0)

def quat_log(q):
    #input: q: Nx4
    #output: Log(q) Nx3 (see Sola eq. 105a/b)
    if q.dim() < 2:
        q = q.unsqueeze(0)

    #Check for negative scalars first, then substitute q for -q whenever that is the case (this accounts for the double cover of S3 over SO(3))
    neg_angle_mask = q[:, 0] < 0.
    neg_angle_inds = neg_angle_mask.nonzero().squeeze_(dim=1)

    q_w = q[:, 0].clone()
    q_v = q[:, 1:].clone()

    if len(neg_angle_inds) > 0:
        q_w[neg_angle_inds] = -1.*q_w[neg_angle_inds]
        q_v[neg_angle_inds] = -1.*q_v[neg_angle_inds]

    q_v_norm = q_v.norm(dim=1)

    # Near phi==0 (q_w ~ 1), use first order Taylor expansion
    angles = 2. * torch.atan2(q_v_norm, q_w)
    small_angle_mask = isclose(angles, 0.)
    small_angle_inds = small_angle_mask.nonzero().squeeze_(dim=1)

    phi = q.new_empty((q.shape[0], 3))



    if len(small_angle_inds) > 0:
        q_v_small = q_v[small_angle_inds]
        q_v_n_small = q_v_norm[small_angle_inds].unsqueeze(1)
        q_w_small = q_w[small_angle_inds].unsqueeze(1)
        phi[small_angle_inds, :] = \
            2. * ( q_v_small /  q_w_small) * \
            (1 - ( q_v_n_small ** 2)/(3. * ( q_w_small ** 2)))


    # Otherwise...
    large_angle_mask = 1 - small_angle_mask  # element-wise not
    large_angle_inds = large_angle_mask.nonzero().squeeze_(dim=1)

    if len(large_angle_inds) > 0:
        angles_large = angles[large_angle_inds]
        #print(q_v[large_angle_inds].shape)
        #print(q_v_norm[large_angle_inds].shape)

        axes = q_v[large_angle_inds] / q_v_norm[large_angle_inds].unsqueeze(1)
        phi[large_angle_inds, :] = \
            angles_large.unsqueeze(1) * axes

    return phi.squeeze()

def quat_inv(q):
    #Note, 'empty_like' is necessary to prevent in-place modification (which is not auto-diff'able)
    if q.dim() < 2:
        q = q.unsqueeze()
    q_inv = torch.empty_like(q)
    q_inv[:, :3] = -1*q[:, :3]
    q_inv[:, 3] = q[:, 3]
    return q_inv.squeeze()

#========================


#Quaternion difference of two unit quaternions
def quat_norm_diff(q_a, q_b):
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
    if q_b.dim() < 2:
        q_b = q_b.unsqueeze(0)
    return torch.min((q_a-q_b).norm(dim=1), (q_a+q_b).norm(dim=1)).squeeze()

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


def quat_to_rotmat(quat, ordering='wxyz'):
    """Form a rotation matrix from a unit length quaternion.

        Valid orderings are 'xyzw' and 'wxyz'.
    """
    if quat.dim() < 2:
        quat = quat.unsqueeze(dim=0)

    if not utils.allclose(quat.norm(p=2, dim=1), 1.):
        raise ValueError("Quaternions must be unit length")

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


def rotmat_to_quat(mat, ordering='wxyz'):
    """Convert a rotation matrix to a unit length quaternion.

        Valid orderings are 'xyzw' and 'wxyz'.
    """
    if mat.dim() < 3:
        R = mat.unsqueeze(dim=0)
    else:
        R = mat

    qw = 0.5 * torch.sqrt(1. + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
    qx = qw.new_empty(qw.shape)
    qy = qw.new_empty(qw.shape)
    qz = qw.new_empty(qw.shape)

    near_zero_mask = utils.isclose(qw, 0.)

    if sum(near_zero_mask) > 0:
        cond1_mask = near_zero_mask & \
            (R[:, 0, 0] > R[:, 1, 1]).squeeze_() & \
            (R[:, 0, 0] > R[:, 2, 2]).squeeze_()
        cond1_inds = cond1_mask.nonzero().squeeze_(dim=1)

        if len(cond1_inds) > 0:
            R_cond1 = R[cond1_inds]
            d = 2. * np.sqrt(1. + R_cond1[:, 0, 0] -
                                R_cond1[:, 1, 1] - R_cond1[:, 2, 2])
            qw[cond1_inds] = (R_cond1[:, 2, 1] - R_cond1[:, 1, 2]) / d
            qx[cond1_inds] = 0.25 * d
            qy[cond1_inds] = (R_cond1[:, 1, 0] + R_cond1[:, 0, 1]) / d
            qz[cond1_inds] = (R_cond1[:, 0, 2] + R_cond1[:, 2, 0]) / d

        cond2_mask = near_zero_mask & (R[:, 1, 1] > R[:, 2, 2]).squeeze_()
        cond2_inds = cond2_mask.nonzero().squeeze_(dim=1)

        if len(cond2_inds) > 0:
            R_cond2 = R[cond2_inds]
            d = 2. * np.sqrt(1. + R_cond2[:, 1, 1] -
                                R_cond2[:, 0, 0] - R_cond2[:, 2, 2])
            qw[cond2_inds] = (R_cond2[:, 0, 2] - R_cond2[:, 2, 0]) / d
            qx[cond2_inds] = (R_cond2[:, 1, 0] + R_cond2[:, 0, 1]) / d
            qy[cond2_inds] = 0.25 * d
            qz[cond2_inds] = (R_cond2[:, 2, 1] + R_cond2[:, 1, 2]) / d

        cond3_mask = near_zero_mask & cond1_mask.logical_not() & cond2_mask.logical_not()
        cond3_inds = cond3_mask.nonzero().squeeze_(dim=1)

        if len(cond3_inds) > 0:
            R_cond3 = R[cond3_inds]
            d = 2. * \
                np.sqrt(1. + R_cond3[:, 2, 2] -
                        R_cond3[:, 0, 0] - R_cond3[:, 1, 1])
            qw[cond3_inds] = (R_cond3[:, 1, 0] - R_cond3[:, 0, 1]) / d
            qx[cond3_inds] = (R_cond3[:, 0, 2] + R_cond3[:, 2, 0]) / d
            qy[cond3_inds] = (R_cond3[:, 2, 1] + R_cond3[:, 1, 2]) / d
            qz[cond3_inds] = 0.25 * d

    far_zero_mask = near_zero_mask.logical_not()
    far_zero_inds = far_zero_mask.nonzero().squeeze_(dim=1)
    if len(far_zero_inds) > 0:
        R_fz = R[far_zero_inds]
        d = 4. * qw[far_zero_inds]
        qx[far_zero_inds] = (R_fz[:, 2, 1] - R_fz[:, 1, 2]) / d
        qy[far_zero_inds] = (R_fz[:, 0, 2] - R_fz[:, 2, 0]) / d
        qz[far_zero_inds] = (R_fz[:, 1, 0] - R_fz[:, 0, 1]) / d

    # Check ordering last
    if ordering is 'xyzw':
        quat = torch.cat([qx.unsqueeze_(dim=1),
                            qy.unsqueeze_(dim=1),
                            qz.unsqueeze_(dim=1),
                            qw.unsqueeze_(dim=1)], dim=1).squeeze_()
    elif ordering is 'wxyz':
        quat = torch.cat([qw.unsqueeze_(dim=1),
                            qx.unsqueeze_(dim=1),
                            qy.unsqueeze_(dim=1),
                            qz.unsqueeze_(dim=1)], dim=1).squeeze_()
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

    return quat

## PYTORCH QUAT LOSSES

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


