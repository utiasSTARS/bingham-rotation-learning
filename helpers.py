import numpy as np
from liegroups import SO3
from numpy.linalg import norm

#Helpers
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

def q_from_qqT(qqT):
    #Returns unit quaternion q from q * q^T 4x4 matrix
    #Assumes scalar is the last value and it is positive (can make this choice since q = -q)

    q = np.sqrt(np.diag(qqT))
    if qqT[0,3] < 0.:
        q[0] *=  -1.
    if qqT[1,3] < 0.:
        q[1] *=  -1.
    if qqT[2,3] < 0.:
        q[2] *=  -1.

    return q

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



def so3_error(C_1, C_2):
    A = SO3.from_matrix(C_1)
    B = SO3.from_matrix(C_2)
    err = A.dot(B.inv()).log()    
    return norm(err)*180./np.pi


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


def make_random_instance(N, N_out, sigma=0.01):
    C = SO3.exp(np.random.randn(3)).as_matrix()
    # Create two sets of vectors (normalized to unit l2 norm)
    x_1 = normalized(np.random.rand(N, 3) - 0.5, axis=1)
    # Rotate and add noise
    x_2 = C.dot(x_1.T).T + sigma * np.random.randn(N, 3)
    # Outliers
    if N_out > 0:
        outlier_indices = np.random.choice(x_2.shape[0], N_out, replace=False)
        x_2[outlier_indices] = 10 * (np.random.rand(N_out, 3) - 0.5)
        outlier_indices = list(outlier_indices)
    else:
        outlier_indices = []
    return x_1, x_2, C, outlier_indices
