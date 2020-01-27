from quaternions import *
import torch
from utils import *
import numpy as np
from liegroups.torch import SO3
from sdp_layers import x_from_xxT
import math
from losses import quat_chordal_squared_loss, rotmat_frob_squared_norm_loss

def test_180_quat():
    a = torch.randn(25,3).to(torch.float64)
    a = a / a.norm(dim=1, keepdim=True)
    angle = (150)*(np.pi/180.)
    aa = a * angle
    C = SO3.exp(aa).as_matrix() 
    print(rotmat_to_quat(C))

def test_rotmat_quat_conversions():
    print('Rotation matrix to quaternion conversions...')
    C1 = SO3.exp(torch.randn(100, 3, dtype=torch.double)).as_matrix()
    C2 = quat_to_rotmat(rotmat_to_quat(C1))
    assert(allclose(C1, C2))
    print('All passed.')


def test_chordal_squared_loss_equality():
    print('Equality of quaternion and rotation matrix chordal loss...')
    C1 = SO3.exp(torch.randn(1000, 3, dtype=torch.double)).as_matrix()
    C2 = SO3.exp(torch.randn(1000, 3, dtype=torch.double)).as_matrix()

    q1 = rotmat_to_quat(C1)
    q2 = rotmat_to_quat(C2)

    assert(allclose(rotmat_frob_squared_norm_loss(C1, C2), quat_chordal_squared_loss(q1, q2)))
    print('All passed.')

def test_rotmat_quat_large_conversions():
    print('Large (angle=pi) rotation matrix to quaternion conversions...')
    axis = torch.randn(100, 3, dtype=torch.double)
    axis = axis / axis.norm(dim=1, keepdim=True)
    angle = np.pi

    C1 = SO3.exp(angle*axis).as_matrix()
    C2_new = quat_to_rotmat(rotmat_to_quat(C1))
    assert(allclose(C1, C2_new))
    print('All passed.')
     
def test_rot_angles():
    print('Rotation angles...')
    C1 = SO3.exp(torch.randn(100, 3, dtype=torch.double))
    C2 = SO3.exp(torch.randn(100, 3, dtype=torch.double))

    angles_1 = (C1.dot(C2.inv())).log().norm(dim=1)*(180./np.pi)
    angles_2 = quat_angle_diff(rotmat_to_quat(C1.as_matrix()), rotmat_to_quat(C2.as_matrix()), units='deg', reduce=False)
    angles_3 = rotmat_angle_diff(C1.as_matrix(), C2.as_matrix(), reduce=False)
    assert(allclose(angles_1, angles_2))
    assert(allclose(angles_1, angles_3))
    print('All passed.')

def test_xxT():
    print('Testing x_from_xxT...')
    x = torch.randn(100, 10, dtype=torch.double)
    x[:,-1] = 1.
    X = outer(x, x)
    assert(allclose(x, x_from_xxT(X)))
    print('All passed.')

if __name__=='__main__':
    # test_rotmat_quat_conversions()
    # test_rot_angles()
    # test_xxT()
    # test_rotmat_quat_large_conversions()
    # test_chordal_squared_loss_equality()
    test_180_quat()