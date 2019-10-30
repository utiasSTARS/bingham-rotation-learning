import numpy as np
from liegroups import SO3
from numpy.linalg import norm
from scipy.linalg import sqrtm
def compute_rotation_from_two_vectors(a_1, a_2, b_1, b_2):
    #Returns C, such that b_1 = C a_1 and b_2 = C a_2
    
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

#Create a random rotation
C = SO3.exp(np.random.randn(3)).as_matrix()

#Create two vectors
sigma = 0.01
a_1 = np.random.rand(3)
a_2 = np.random.rand(3)
b_1 = C.dot(a_1) + sigma*np.random.randn(3)
b_2 = C.dot(a_2) + sigma*np.random.randn(3)

C_est = compute_rotation_from_two_vectors(a_1, a_2, b_1, b_2)

#Normalize matrix
C_est = np.linalg.inv(sqrtm(C_est.dot(C_est.T))).dot(C_est)
#C_est2 = SO3.from_matrix(C_est, normalize=True).as_matrix()

#Compares
# print(b_1)
# print(C_est.dot(a_1))
# print(b_2)
# print(C_est.dot(a_2))
# print('')
print(np.linalg.det(C_est))
print(np.linalg.norm(C-C_est))
