import numpy as np
from scipy.stats import chi2
from numpy.linalg import norm 
from liegroups.numpy import SO3
import matplotlib.pylab as plt
import cvxpy as cp
import time


np.random.seed(42)
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

######

##Parameters

#Sim
N = 5
sigma = 0 #0.01
N_out = 0 #How many of N samples are outliers

#Solver
sigma_2_i = 1 #sigma**2#(100*sigma)**2
p_false_negative = 0.0001 # Probability an inlier is classified as an outlier
#c_bar_2 = sigma**2*chi2.ppf(1-p_false_negative, df=3)
c_bar_2 = 0.1
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
    x_2[np.random.randint(0,x_2.shape[0], N_out)] = 5*np.random.rand(N_out, 3)
    
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

#Optional: visualize matrix
# plt.spy(Q)
# plt.show()

#Build Z variable with constraints
Z = cp.Variable((4*(N+1),4*(N+1)), symmetric=True)
constraints = [Z >> 0]

#Naive constraints
constraints += [
    cp.trace(Z[:4,:4]) == 1
]
constraints += [
    Z[(i)*4:(i+1)*4, (i)*4:(i+1)*4] == Z[:4,:4] for i in range(1, N)
]

#Additional non-naive constraints
if redundant_constraints:
    #q q_i
    constraints += [
        Z[:4, (i)*4:(i+1)*4] == Z[:4, (i)*4:(i+1)*4].T for i in range(1, N)
    ]

    # q_i q_j
    for i in range(2,N):
        constraints += [
            Z[4*i:4*(i+1), (j)*4:(j+1)*4] == Z[4*i:4*(i+1), (j)*4:(j+1)*4].T for j in range(i+1, N)
        ]

#Solve SDP
print('Set up problem data and constraints')
print('Solving..')
start_time = time.time()

prob = cp.Problem(cp.Minimize(cp.trace(Q@Z)),
                  constraints)
prob.solve()
print("Final status:", prob.status)
#print(Z.value)
q_est = q_from_qqT(Z.value[:4,:4])
C_est = SO3.from_quaternion(q_est, ordering='xyzw').as_matrix()
end_time = time.time()
print('Done. Solved in {:.3f} seconds.'.format(end_time - start_time))

#Compare to known rotation
# C_est = C
print('SO(3) Frob norm error:')
print(np.linalg.norm(C-C_est))
