import numpy as np
import autograd.numpy as anp
from helpers import *
from pymanopt.manifolds import PSDFixedRank, PSDFixedRankComplex
from scipy.stats import chi2
from numpy.linalg import norm 
from liegroups.numpy import SO3
import matplotlib.pylab as plt
import time
from pymanopt.manifolds import PSDFixedRank
from pymanopt import Problem
from pymanopt.solvers import TrustRegions

##Parameters

#Sim
N = 4
sigma = 0.001 #0.01
N_out = 0 #How many of N samples are outliers

#SDP seems to react poorly to small values of sigma_2_i
#Instead, we rely on the interpretation that we reject any residual larger than sigma_i*c_bar as an outlier
#Set sigma_i to 1, set c_bar to 3*sigma     
sigma_2_i = 1
c_bar_2 = (3*(sigma+1e-3))**2 
print('c_bar_2: {:.3f}'.format(c_bar_2))
redundant_constraints = True


##Simulation
#Create a random rotation
C = SO3.exp(np.random.randn(3)).as_matrix()

#Create two sets of vectors (normalized to unit l2 norm)
x_1 = normalized(np.random.rand(N, 3) - 0.5, axis=1)
#Rotate and add noise
x_2 = C.dot(x_1.T).T + sigma*np.random.randn(N,3)

#Outliers
if N_out > 0:
    x_2[np.random.choice(x_2.shape[0], N_out, replace=False)] = 5*np.random.rand(N_out, 3)
    
## Solver
#Build Q matrix
#No sparsity for now
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

manifold = PSDFixedRank(4*(N+1), 1) #Rank 1


def cost(Z):
    return anp.trace(
            anp.dot(
                Q, anp.dot(Z, anp.transpose(Z))
            )
        ) 


#Solve SDP
print('Set up problem data')

solver = TrustRegions()
problem = Problem(manifold=manifold, cost=cost)

print('Solving..')
start_time = time.time()

z_opt = solver.solve(problem)

print(z_opt[:4])
q_est =  z_opt[:4]/norm(z_opt[:4])

C_est = SO3.from_quaternion(q_est, ordering='xyzw').as_matrix()
end_time = time.time()
print('Done. Solved in {:.3f} seconds.'.format(end_time - start_time))

#Compare to known rotation
# C_est = C
print('SO(3) Frob norm error:')
print(np.linalg.norm(C-C_est))
