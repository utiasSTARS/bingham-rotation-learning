import numpy as np
from scipy.stats import chi2
from numpy.linalg import norm 
from liegroups.numpy import SO3
import matplotlib.pylab as plt
import cvxpy as cp
import time
from helpers import *
from quasar import *
import matplotlib.pyplot as plt
                

##Parameters

#Sim
N = 100 #Numbers of vectors in each cloud
#N_ransac = 100 #How many times do we sample?
N_batch = 3
p = 0.99
alpha = 5 #Factor that adjusts probability of sampling inlier correspondence. alpha = 1 corresponds to chance (i.e., 1/N)
N_qransac = np.int(np.log(1 - p)/np.log((1.-alpha/N)**N_batch + N_batch*(alpha/N)*(1.-alpha/N)**(N_batch-1)))
sigma = 0.01 #Noise
redundant_constraints = True

#Solver
# sigma_2_i = sigma**2
#p_false_negative = 0.001 # Probability an inlier is classified as an outlier
#c_bar_2 = chi2.ppf(1-p_false_negative, df=3)

#SDP seems to react poorly to small values of sigma_2_i
#Instead, we rely on the interpretation that we reject any residual larger than sigma_i*c_bar as an outlier
#Set sigma_i to 1, set c_bar to 3*sigma     
sigma_2_i = 1
c_bar_2 = ((3*sigma))**2 
print('c_bar_2: {:.3f}'.format(c_bar_2))


##Simulation
#Create a random rotation
C = SO3.exp(np.random.randn(3)).as_matrix()

#Create two sets of vectors (normalized to unit l2 norm)
x_1 = normalized(10*(np.random.rand(N, 3) - 0.5), axis=1)
#Rotate and add noise
x_2 = C.dot(x_1.T).T + sigma*np.random.randn(N,3)

start_time = time.time()


#map N*N correspondeces to [0, N*N-1]

#True inliers
true_match_idx = (N+1)*np.arange(0, N)

#Create sampling probabilities that make it alpha times more likely to select inliers
sample_probs = np.ones(N*N)
sample_probs[true_match_idx] *= alpha
sample_probs = sample_probs/np.sum(sample_probs)

max_inliers = 0

gaps = np.empty(N_qransac)
true_in = np.empty(N_qransac)


for i in range(N_qransac):

    #This has an optional 'p' parameter that can weight correspondences
    #Correct correspondences are at the indices i*(1+N) for i = 0...N-1
    sample_match_idx = np.random.choice(N*N, N_batch, replace=False, p = sample_probs)
    


    index_2 = np.mod(sample_match_idx, N).astype(int)
    index_1 = np.floor((sample_match_idx - index_2)/N).astype(int)

    x_1_select = x_1[index_1]
    x_2_select = x_2[index_2]



    q_est, est_outlier_indices, t_solve, gap, obj_cost = solve_quasar(x_1_select, x_2_select, c_bar_2,
                                                                                      redundant_constraints=redundant_constraints)

    C_est = SO3.from_quaternion(q_est, ordering='xyzw').as_matrix()    
    #Inliers over the entire set
    inlier_id_1, inlier_id_2 = compute_inlier_matches(x_1, x_2, C_est, c_bar_2, sigma_2_i)
    num_total_inliers = inlier_id_1.shape[0]
    num_true_inliers = np.sum(index_1==index_2)

    print('{} gap -> {} true inliers'.format(gap, num_true_inliers))
    gaps[i] = gap
    true_in[i] = num_true_inliers
    # print('Number of large eigenvalues: {}'.format(num_large_eigs))
    # if num_large_eigs == 1:
    #   print(eigs)

    #true_matches = np.intersect1d(sample_match_idx, true_match_idx)

    #if true_matches.shape[0] > 1:
    #print('Found {} true matches.'.format(true_matches.shape[0]))

    if num_total_inliers > max_inliers: #rank_1_ratio > 8:
        #Compute all inliers
        print('QUASAR found new best model.')
        print('--> {} inliers.'.format(num_total_inliers))

        
        #print('QUASAR SO(3) err: {:.3f} deg'. format(so3_error(C, C_est)))
        #print('Using QUASAR solution to find inliers')

        max_inliers = num_total_inliers
        best_inliers = [inlier_id_1, inlier_id_2]

        if max_inliers > 0.8*N:
            print('Early stopping!')
            break
        # C_test = compute_rotation_from_two_vectors(x_1_select[inlier_idx[0]], x_1_select[inlier_idx[1]], 
        #                                             x_2_select[inlier_idx[0]], x_2_select[inlier_idx[1]])
        # print(np.linalg.norm(C_test-C))

        #print('All inlier SO(3) (Horn method) err: {:.3f} deg'. format(so3_error(C, C_horn)))
        
    print('{}/{}'.format(i+1, N_qransac))

C_horn = solve_horn(x_1[best_inliers[0]], x_2[best_inliers[1]])

end_time = time.time()

print('Done. Solved in {:.3f} seconds.'.format(end_time - start_time))
print('All inlier SO(3) (Horn method) err: {:.3f} deg'. format(so3_error(C, C_horn)))
# plt.semilogy(true_in, gaps, 'bo')
# plt.show()