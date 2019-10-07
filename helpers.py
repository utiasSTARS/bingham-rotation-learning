import numpy as np
from liegroups import SO3
from numpy.linalg import norm

#Helpers
##########
def Omega_1(q):
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

def Omega_2(q):
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

def Q_ii(a_i, b_i, c_bar_2, sigma_2_i):
    I = np.eye(4)
    t1 = (b_i.dot(b_i) + a_i.dot(a_i))*I
    t2 = 2*Omega_1(pure_quat(b_i)).dot(
        Omega_2(pure_quat(a_i))
        )
    Q = (t1 + t2)/(2*sigma_2_i) + 0.5*c_bar_2*I
    return Q

def Q_0i(a_i, b_i, c_bar_2, sigma_2_i):
    I = np.eye(4)
    t1 = (b_i.dot(b_i) + a_i.dot(a_i))*I
    t2 = 2*Omega_1(pure_quat(b_i)).dot(
        Omega_2(pure_quat(a_i))
        )
    Q = (t1 + t2)/(4*sigma_2_i) - 0.25*c_bar_2*I
    return Q


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

def extract_outlier_indices(qqT):
    outlier_indices = []
    inlier_indices = []
    N = round(qqT.shape[0]/4) - 1
    for i in range(1, N+1):
        qqiT = qqT[:4, 4*i:4*(i+1)]
        if qqiT[0,0] < 0:
            outlier_indices.append(i-1) 
        else:
            inlier_indices.append(i-1)
    return np.array(outlier_indices), np.array(inlier_indices)



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


def compute_inlier_matches(x_1, x_2, C_est, c_bar_2, sigma_2_i):

    N = x_1.shape[0]
    inlier_1_idx = []
    inlier_2_idx = []
    
    for i in range(N):
        for j in range(N):
            err = C_est.dot(x_1[i].T).T - x_2[j]
            if err.dot(err)/sigma_2_i < c_bar_2:
                inlier_1_idx.append(i)
                inlier_2_idx.append(j)
    
    return np.array(inlier_1_idx), np.array(inlier_2_idx)


def build_cost_function_matrix(x_1, x_2, c_bar_2, sigma_2_i):
    N = x_1.shape[0]
    Q = np.zeros((4 * (N + 1), 4 * (N + 1)))
    # for i in range(N):
    for ii in range(N):
        Q_i = np.zeros((4 * (N + 1), 4 * (N + 1)))
        # Block diagonal indices
        idx_range = slice((ii + 1) * 4, (ii + 2) * 4)
        Q_i[idx_range, idx_range] = Q_ii(x_1[ii], x_2[ii], c_bar_2, sigma_2_i)
        Q_0ii = Q_0i(x_1[ii], x_2[ii], c_bar_2, sigma_2_i)
        Q_i[:4, idx_range] = Q_0ii
        Q_i[idx_range, :4] = Q_0ii
        Q += Q_i

    return Q


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
