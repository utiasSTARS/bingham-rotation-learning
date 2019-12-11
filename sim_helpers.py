import numpy as np
import torch
from liegroups.numpy import SO3
from numpy.linalg import norm
from quaternions import *


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

def matrix_diff(X,Y):
    return np.abs(np.linalg.norm(X - Y) / min(np.linalg.norm(X), np.linalg.norm(Y)))
    
def build_A(x_1, x_2, sigma_2):
    N = x_1.shape[0]
    A = np.zeros((4, 4), dtype=np.float64)
    for i in range(N):
        # Block diagonal indices
        I = np.eye(4, dtype=np.float64)
        t1 = (x_2[i].dot(x_2[i]) + x_1[i].dot(x_1[i]))*I
        t2 = 2.*Omega_l(pure_quat(x_2[i])).dot(
            Omega_r(pure_quat(x_1[i])))
        A_i = (t1 + t2)/(sigma_2[i])
        A += A_i
    return A 
#Note sigma can be scalar or an N-dimensional vector of std. devs.
def gen_sim_data(N, sigma, torch_vars=False, shuffle_points=False):
    ##Simulation
    #Create a random rotation
    C = SO3.exp(np.random.randn(3)).as_matrix()
    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = normalized(np.random.randn(N, 3), axis=1)
    #Rotate and add noise
    noise = np.random.randn(N,3)
    noise = (noise.T*sigma).T
    x_2 = C.dot(x_1.T).T + noise

    if shuffle_points:
        x_1, x_2 = unison_shuffled_copies(x_1,x_2)

    if torch_vars:
        C = torch.from_numpy(C)
        x_1 = torch.from_numpy(x_1)
        x_2 = torch.from_numpy(x_2)

    return C, x_1, x_2

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def gen_sim_data_grid(N, sigma, torch_vars=False, shuffle_points=False):
    ##Simulation
    #Create a random rotation
    C = SO3.exp(np.random.randn(3)).as_matrix()
    
    #Grid is fixed 
    grid_dim = 50
    xlims = np.linspace(-1., 1., grid_dim)
    ylims = np.linspace(-1., 1., grid_dim)
    x, y = np.meshgrid(xlims, ylims)
    z = np.sin(x)*np.cos(y)
    x_1 =  normalized(np.hstack((x.reshape(grid_dim**2, 1), y.reshape(grid_dim**2, 1), z.reshape(grid_dim**2, 1))), axis=1)
    
    #Sample N points
    ids = np.random.permutation(x_1.shape[0])
    x_1 = x_1[ids[:N]]

    #Sort into canonical order
    #x_1 = x_1[x_1[:,0].argsort()]

    #Rotate and add noise
    noise = np.random.randn(N,3)
    noise = (noise.T*sigma).T
    x_2 = C.dot(x_1.T).T + noise

    if shuffle_points:
        x_1, x_2 = unison_shuffled_copies(x_1,x_2)


    if torch_vars:
        C = torch.from_numpy(C)
        x_1 = torch.from_numpy(x_1)
        x_2 = torch.from_numpy(x_2)

    return C, x_1, x_2

class SyntheticData():
    def __init__(self, x, q, A_prior):
        self.x = x
        self.q = q
        self.A_prior = A_prior


def create_experimental_data(N_train=2000, N_test=50, N_matches_per_sample=100, sigma=0.01, dtype=torch.double):

    x_train = torch.empty(N_train, 2, N_matches_per_sample, 3, dtype=dtype)
    q_train = torch.empty(N_train, 4, dtype=dtype)
    A_prior_train = torch.empty(N_train, 4, 4, dtype=dtype)

    x_test = torch.empty(N_test, 2, N_matches_per_sample, 3, dtype=dtype)
    q_test = torch.empty(N_test, 4, dtype=dtype)
    A_prior_test = torch.empty(N_test, 4, 4, dtype=dtype)

    sigma_sim_vec = sigma*np.ones(N_matches_per_sample)
    #sigma_sim_vec[:int(N_matches_per_sample/2)] *= 10 #Artificially scale half the noise
    sigma_prior_vec = sigma*np.ones(N_matches_per_sample)
    

    for n in range(N_train):

        C, x_1, x_2 = gen_sim_data(N_matches_per_sample, sigma_sim_vec, torch_vars=True, shuffle_points=False)
        q = rotmat_to_quat(C, ordering='xyzw')
        x_train[n, 0, :, :] = x_1
        x_train[n, 1, :, :] = x_2
        q_train[n] = q
        A_prior_train[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))

    for n in range(N_test):
        C, x_1, x_2 = gen_sim_data(N_matches_per_sample, sigma_sim_vec, torch_vars=True, shuffle_points=False)
        q = rotmat_to_quat(C, ordering='xyzw')
        x_test[n, 0, :, :] = x_1
        x_test[n, 1, :, :] = x_2
        q_test[n] = q
        A_prior_test[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))

        # A_vec = convert_A_to_Avec(A_prior_test[n]).unsqueeze(dim=0)
        # print(q - QuadQuatFastSolver.apply(A_vec).squeeze())
    
    train_data = SyntheticData(x_train, q_train, A_prior_train)
    test_data = SyntheticData(x_test, q_test, A_prior_test)
    
    return train_data, test_data


def compute_mean_horn_error(sim_data):
    N = sim_data.x.shape[0]
    err = torch.empty(N)
    for i in range(N):
        x = sim_data.x[i]
        x_1 = x[0,:,:].numpy()
        x_2 = x[1,:,:].numpy()
        C = torch.from_numpy(solve_horn(x_1, x_2))
        q_est = rotmat_to_quat(C, ordering='xyzw')
        err[i] = quat_angle_diff(q_est, sim_data.q[i])
    return err.mean()

