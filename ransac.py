import numpy as np
from liegroups.numpy import SO3
from helpers import *

#Sim
N = 40
sigma = 0.01 #0.01
N_out = 20 #How many of N samples are outliers


##Simulation
#Create a random rotation
C = SO3.exp(np.random.randn(3)).as_matrix()

#Create two sets of vectors (normalized to unit l2 norm)
x_1 = normalized(np.random.rand(N, 3) - 0.5, axis=1)
#Rotate and add noise
x_2 = C.dot(x_1.T).T + sigma*np.random.randn(N,3)

#Outliers
if N_out > 0:
    outlier_indices = np.random.choice(x_2.shape[0], N_out, replace=False)
    x_2[outlier_indices] = 10*(np.random.rand(N_out, 3) - 0.5)


p = 0.999
w = 1 - N_out/N
n = 2
thresh_ransac = 0.5*(np.pi/180.) #on the vector angle in radians
N_ransac = np.int(np.log(1 - p)/np.log(1 - w**n))

print('Using {} RANSAC iterations.'.format(N_ransac))
max_inliers = 0
for r_i in range(N_ransac):
    
    #Select two correspondences at random
    random_indices = np.random.choice(x_1.shape[0], 2, replace=False)
    a_1 = x_1[random_indices[0]]
    a_2 = x_1[random_indices[1]]
    b_1 = x_2[random_indices[0]]
    b_2 = x_2[random_indices[1]]

    #Compute model
    C_test = compute_rotation_from_two_vectors(a_1, a_2, b_1, b_2)
    #Count inliers
    x_2_test = C_test.dot(x_1.T).T

    dot_products = np.sum(normalized(x_2_test) * normalized(x_2), axis=1)
    dot_products[dot_products>1.] = 1.

    inlier_mask = np.arccos(dot_products) < thresh_ransac
    num_inliers = np.sum(inlier_mask)

    #Track maximum number of inliers
    if num_inliers > max_inliers:
        max_inliers = num_inliers
        best_inlier_indices, = np.nonzero(inlier_mask)
        C_best = C_test

print('Number of inliers found: {}'.format(max_inliers))
print('RANSAC inliers: {}'.format(best_inlier_indices.tolist()))

x_temp = x_2[:,0].copy()
x_temp[outlier_indices] = 0
true_inlier_indices, = np.nonzero(x_temp)
print('True inliers: {}'.format(true_inlier_indices.tolist()))
print('SO(3) Frobenius error: {:.3f}'.format(np.linalg.norm(C-C_best)))