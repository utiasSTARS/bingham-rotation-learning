from helpers_sim import *
import numpy as np
from liegroups.numpy import *
from utils import *
from rotation_matrix_sdp import *

def check_eigs():
    N = 100
    sigma = np.exp(np.random.rand())
    C = SO3.exp(np.random.randn(3)).as_matrix()
    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = normalized(np.random.randn(N, 3), axis=1)
    #Rotate and add noise
    noise = np.random.randn(N,3)
    noise = (noise.T*sigma).T
    x_2 = C.dot(x_1.T).T + noise
    A = build_A(x_1, x_2, sigma*sigma*np.ones(N))
    A = A / np.linalg.norm(A)
    eigs = np.linalg.eigvals(A)
    C_est = solve_horn(x_1, x_2)

    print(rotmat_angle_diff(torch.from_numpy(C_est), torch.from_numpy(C), units='rad'))
    print(0.1/np.log(np.max(eigs)/np.min(eigs)))


    
    


if __name__=='__main__':
    check_eigs()
