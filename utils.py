import torch
import numpy as np
from liegroups.numpy import SO3
from numpy.linalg import norm
import math

def allclose(mat1, mat2, tol=1e-6):
    """Check if all elements of two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return isclose(mat1, mat2, tol).all()


def isclose(mat1, mat2, tol=1e-6):
    """Check element-wise if two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return (mat1 - mat2).abs_().lt(tol)


def outer(vecs1, vecs2):
    """Return the N x D x D outer products of a N x D batch of vectors,
    or return the D x D outer product of two D-dimensional vectors.
    """
    # Default batch size is 1
    if vecs1.dim() < 2:
        vecs1 = vecs1.unsqueeze(dim=0)

    if vecs2.dim() < 2:
        vecs2 = vecs2.unsqueeze(dim=0)

    if vecs1.shape[0] != vecs2.shape[0]:
        raise ValueError("Got inconsistent batch sizes {} and {}".format(
            vecs1.shape[0], vecs2.shape[0]))

    return torch.bmm(vecs1.unsqueeze(dim=2),
                     vecs2.unsqueeze(dim=2).transpose(2, 1)).squeeze_()


def trace(mat):
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    tr =  (torch.eye(mat.shape[1], dtype=mat.dtype) * mat).sum(dim=1).sum(dim=1)
    
    return tr.view(mat.shape[0])


# N x 3 -> N x 3 (unit norm)
def normalize_vectors(vecs):
    if vecs.dim() < 2:
        vecs = vecs.unsqueeze(dim=0)
    return vecs/vecs.norm(dim=1, keepdim=True, p=2)
    
# N x 3, N x 3 -> N x 3 (cross product)
def cross_product(u, v):
    assert(u.dim() == v.dim())
    if u.dim() < 2:
        u = u.unsqueeze(dim=0)
        v = v.unsqueeze(dim=0)
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    return torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)
        
# N x 6 -> N x 3 x 3 (rotation matrices)
# @inproceedings{zhou_continuity_2019,
#   title = {On the {{Continuity}} of {{Rotation Representations}} in {{Neural Networks}}},
#   booktitle = CVPR,
#   author = {Zhou, Yi and Barnes, Connelly and Lu, Jingwan and Yang, Jimei and Li, Hao},
#   year = {2019},
#   pages = {9}
# }

def sixdim_to_rotmat(sixdim):
    if sixdim.dim() < 2:
        sixdim = sixdim.unsqueeze(dim=0)
    x_raw = sixdim[:,0:3]#batch*3
    y_raw = sixdim[:,3:6]#batch*3
        
    x = normalize_vectors(x_raw)
    z = cross_product(x,y_raw) 
    z = normalize_vectors(z)
    y = cross_product(z,x)
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    rotmat = torch.cat((x,y,z), 2) #batch*3*3
    return rotmat

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def compute_rotation_from_two_vectors(a_1, a_2, b_1, b_2):
    #Returns C in SO(3), such that b_1 = C*a_1 and b_2 = C*a_2
    
    ## Construct orthonormal basis of 'a' frame
    a_1_u = a_1/(norm(a_1))
    a_2_u = a_2/(norm(a_2))
    alpha = a_1_u.dot(a_2_u)

    a_basis_1 = a_1_u
    a_basis_2 = a_2_u - alpha*a_1_u
    a_basis_2 /= norm(a_basis_2)
    a_basis_3 = np.cross(a_basis_1, a_basis_2)

    ## Construct basis of 'b' frame
    b_basis_1 = b_1/norm(b_1)
    b_basis_2 = b_2/norm(b_2) - alpha*b_basis_1
    b_basis_2 /= norm(b_basis_2)
    b_basis_3 = np.cross(b_basis_1, b_basis_2)


    #Basis of 'a' frame as column vectors
    M_a = np.array([a_basis_1, a_basis_2, a_basis_3])

    #Basis of 'b' frame as row vectors
    M_b = np.array([b_basis_1, b_basis_2, b_basis_3]).T


    #Direction cosine matrix from a to b!
    C = M_b.dot(M_a)
    
    return C

def so3_diff(C_1, C_2, unit='deg'):
    A = SO3.from_matrix(C_1)
    B = SO3.from_matrix(C_2)
    err = A.dot(B.inv()).log()
    if unit=='deg':
        return norm(err)*180./np.pi
    else:
        return norm(err)
        

def solve_horn(x_1, x_2):
    x_1 = normalized(x_1, axis=1)
    x_2 = normalized(x_2, axis=1)
    
    if x_1.shape[0] == 2:
        
        x_1 = np.append(x_1, np.expand_dims(np.cross(x_1[0], x_1[1]), axis=0), axis=0)
        x_2 = np.append(x_2, np.expand_dims(np.cross(x_2[0], x_2[1]), axis=0), axis=0)

    x_1_n = x_1 - np.mean(x_1, axis=0)
    x_2_n = x_2 - np.mean(x_2, axis=0)
    
    W = (1./(x_1.shape[0]))*x_2_n.T.dot(x_1_n)

    U,_,V = np.linalg.svd(W, full_matrices=False)
    S = np.eye(3)
    S[2,2] = np.linalg.det(U) * np.linalg.det(V)
    C = U.dot(S).dot(V)
    return C

def solve_horn_batch(x_1, x_2):
    """Assumes x_1, x_2 are BXNX3, outputs Bx3x3 matrices"""
    

    x_1_n = x_1 - x_1.mean(dim=1, keepdim=True)
    x_2_n = x_2 - x_1.mean(dim=1, keepdim=True)
    
    W = (1./(x_1.shape[0]))*x_2_n.T.dot(x_1_n)

    U,_,V = torch.svd(W)
    S = np.eye(3)
    S[2,2] = np.linalg.det(U) * np.linalg.det(V)
    C = U.dot(S).dot(V)
    return C

def matrix_diff(X,Y):
    return np.abs(np.linalg.norm(X - Y) / min(np.linalg.norm(X), np.linalg.norm(Y)))
    
def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))