import numpy as np
from scipy.stats import chi2
from numpy.linalg import norm 
from liegroups.numpy import SO3
import matplotlib.pylab as plt
import cvxpy as cp
import time
from helpers import *


                  
def solve_sdp(x_1, x_2, c_bar_2, sigma_2_i, redundant_constraints=True):
    N = x_1.shape[0]
    assert(x_1.shape == x_2.shape)

    Q = np.zeros((4*(N+1), 4*(N+1)))

    # for i in range(N):
    for ii in range(N):
        Q_i = np.zeros((4*(N+1), 4*(N+1)))
        #Block diagonal indices
        idx_range = slice( (ii+1)*4 , (ii+2)*4 )
        Q_i[idx_range, idx_range] = Q_ii(x_1[ii], x_2[ii], c_bar_2, sigma_2_i)
        Q_0ii =  Q_0i(x_1[ii], x_2[ii], c_bar_2, sigma_2_i)
        Q_i[:4, idx_range] = Q_0ii
        Q_i[idx_range, :4] = Q_0ii

        Q += Q_i

    #Build Z variable with constraints
    Z = cp.Variable((4*(N+1),4*(N+1)), PSD=True)

    #Naive constraints
    constraints = [
        cp.trace(Z[:4,:4]) == 1
    ]
    constraints += [
        Z[(i)*4:(i+1)*4, (i)*4:(i+1)*4] == Z[:4,:4] for i in range(1, N+1)
    ]

    #Additional non-naive constraints
    if redundant_constraints:
        #q q_i
        constraints += [
            Z[:4, (i)*4:(i+1)*4] == Z[:4, (i)*4:(i+1)*4].T for i in range(1, N+1)
        ]

        # q_i q_j
        for i in range(1,N+1):
            constraints += [
                Z[4*i:4*(i+1), (j)*4:(j+1)*4] == Z[4*i:4*(i+1), (j)*4:(j+1)*4].T for j in range(i+1, N+1)
            ]

    prob = cp.Problem(cp.Minimize(cp.trace(Q@Z)),
                    constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    eigs = np.linalg.eigvals(Z.value)
    #Extract outliers
    outlier_idx, inlier_idx = extract_outlier_indices(Z.value)
    q_est = q_from_qqT(Z.value[:4,:4])
    C_est = SO3.from_quaternion(q_est, ordering='xyzw').as_matrix()

    return C_est, outlier_idx, inlier_idx, eigs

##Parameters

#Sim
N = 100 #Numbers of vectors in each cloud
N_ransac = 100 #How many times do we sample?
N_batch = 5
sigma = 0.01 #Noise
redundant_constraints = False
alpha = 5 #Factor that adjusts probability of sampling inlier correspondence. alpha = 1 corresponds to chance (i.e., 1/N^2)

#Solver
# sigma_2_i = sigma**2
#p_false_negative = 0.001 # Probability an inlier is classified as an outlier
#c_bar_2 = chi2.ppf(1-p_false_negative, df=3)

#SDP seems to react poorly to small values of sigma_2_i
#Instead, we rely on the interpretation that we reject any residual larger than sigma_i*c_bar as an outlier
#Set sigma_i to 1, set c_bar to 3*sigma     
sigma_2_i = 1
c_bar_2 = ((3*sigma+1e-4))**2 
print('c_bar_2: {:.3f}'.format(c_bar_2))
redundant_constraints = True


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
true_match_idx = (N+1)*np.arange(0, N-1)
#Create sampling probabilities that make it alpha times more likely to select inliers


sample_probs = np.ones(N*N)
sample_probs[true_match_idx] *= alpha
sample_probs = sample_probs/np.sum(sample_probs)

max_inliers = 0
for i in range(N_ransac):

    #This has an optional 'p' parameter that can weight correspondences
    #Correct correspondences are at the indices i*(1+N) for i = 0...N-1

    sample_match_idx = np.random.choice(N*N, N_batch, replace=False, p = sample_probs)
    


    index_1 = np.floor(sample_match_idx/N)
    index_2 = sample_match_idx - N*index_1

    x_1_select = x_1[index_1.astype(int)]
    x_2_select = x_2[index_2.astype(int)]


    C_est, outlier_idx, inlier_idx, eigs = solve_sdp(x_1_select, x_2_select, c_bar_2, sigma_2_i, redundant_constraints=redundant_constraints)
    
    #Inliers over the entire set
    inlier_id_1, inlier_id_2 = compute_inlier_matches(x_1, x_2, C_est, c_bar_2, sigma_2_i)
    num_total_inliers = inlier_id_1.shape[0]
    # print('Number of large eigenvalues: {}'.format(num_large_eigs))
    # if num_large_eigs == 1:
    #   print(eigs)

    true_matches = np.intersect1d(sample_match_idx, true_match_idx)

    #if true_matches.shape[0] > 1:
    #print('Found {} true matches.'.format(true_matches.shape[0]))

    # Rank 1 solution
    rank_1_ratio = np.log10(eigs[0]/eigs[1]) 

    if num_total_inliers > max_inliers: #rank_1_ratio > 8:
        #Compute all inliers
        print('QUASAR found new best model.')
        print('--> {} inliers.'.format(num_total_inliers))
        
        #print('QUASAR SO(3) err: {:.3f} deg'. format(so3_error(C, C_est)))
        #print('Using QUASAR solution to find inliers')

        C_horn = solve_horn(x_1[inlier_id_1], x_2[inlier_id_2])
        C_best = C_horn
        max_inliers = num_total_inliers
        # C_test = compute_rotation_from_two_vectors(x_1_select[inlier_idx[0]], x_1_select[inlier_idx[1]], 
        #                                             x_2_select[inlier_idx[0]], x_2_select[inlier_idx[1]])
        # print(np.linalg.norm(C_test-C))

        #print('All inlier SO(3) (Horn method) err: {:.3f} deg'. format(so3_error(C, C_horn)))
        
    print('{}/{}'.format(i+1, N_ransac))

end_time = time.time()
print('Done. Solved in {:.3f} seconds.'.format(end_time - start_time))
print('All inlier SO(3) (Horn method) err: {:.3f} deg'. format(so3_error(C, C_best)))
        