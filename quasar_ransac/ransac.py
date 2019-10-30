import numpy as np
from liegroups.numpy import SO3
from helpers import *
import time

#Sim
N = 100
sigma = 0.01

##Simulation
#Create a random rotation
C = SO3.exp(np.random.randn(3)).as_matrix()

# #Create two sets of vectors (normalized to unit l2 norm)
# x_1 = normalized(np.random.rand(N, 3) - 0.5, axis=1)
# #Rotate and add noise
# x_2 = C.dot(x_1.T).T + sigma*np.random.randn(N,3)

# #Outliers
# if N_out > 0:
#     outlier_indices = np.random.choice(x_2.shape[0], N_out, replace=False)
#     x_2[outlier_indices] = 10*(np.random.rand(N_out, 3) - 0.5)

#Create two sets of vectors (normalized to unit l2 norm)
x_1 = normalized(10*(np.random.rand(N, 3) - 0.5), axis=1)
#Rotate and add noise
x_2 = C.dot(x_1.T).T + sigma*np.random.randn(N,3)


p = 0.99
alpha = 1
N_batch = 2
w = alpha/N

sigma_2_i = 1
c_bar_2 = ((3*sigma))**2 
N_ransac = np.int(np.log(1 - p)/np.log(1 - w**N_batch))

print('Using {} RANSAC iterations.'.format(N_ransac))
max_inliers = 0
start_time = time.time()


true_match_idx = (N+1)*np.arange(0, N)
sample_probs = np.ones(N*N)
sample_probs[true_match_idx] *= alpha
sample_probs = sample_probs/np.sum(sample_probs)

best_inliers = []
for r_i in range(N_ransac):
    #This has an optional 'p' parameter that can weight correspondences
    #Correct correspondences are at the indices i*(1+N) for i = 0...N-1
    sample_match_idx = np.random.choice(N*N, N_batch, replace=False, p = sample_probs)
    index_2 = np.mod(sample_match_idx, N).astype(int)
    index_1 = np.floor((sample_match_idx - index_2)/N).astype(int)
    x_1_select = x_1[index_1]
    x_2_select = x_2[index_2]

    # if (index_1 == index_2).all():
    #     print('Selected INLIERS!')
    #     print(index_1)
    #     print(index_2)
        
    
    
    
    # #Select two correspondences at random
    # random_indices = np.random.choice(x_1.shape[0], 2, replace=False)
    # a_1 = x_1[random_indices[0]]
    # a_2 = x_1[random_indices[1]]
    # b_1 = x_2[random_indices[0]]
    # b_2 = x_2[random_indices[1]]

    #Compute model
    C_est = compute_rotation_from_two_vectors(x_1_select[0], x_1_select[1], x_2_select[0], x_2_select[1])
    #C_est = solve_horn(x_1_select, x_2_select)

    #Count inliers
    inlier_id_1, inlier_id_2 = compute_inlier_matches(x_1, x_2, C_est, c_bar_2, sigma_2_i)
    num_total_inliers = inlier_id_1.shape[0]

   
    #Track maximum number of inliers
    if num_total_inliers > max_inliers:
        print('RANSAC found new best model.')
        print('--> {} inliers.'.format(num_total_inliers))

        max_inliers = num_total_inliers
        best_inliers = [inlier_id_1, inlier_id_2]
        if max_inliers > 0.8*N:
            print('Early stopping!')
            break
    
    print('{}/{}'.format(r_i+1, N_ransac))

end_time = time.time()
print('Done. Solved in {:.3f} seconds.'.format(end_time - start_time))
if best_inliers == []:
    print('No inliers found.')
else:
    C_horn = solve_horn(x_1[best_inliers[0]], x_2[best_inliers[1]])
    print('All inlier SO(3) (Horn method) err: {:.3f} deg'. format(so3_error(C, C_horn)))