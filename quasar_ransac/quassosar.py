# QUaternion-based Sparse Sum Of Squares relAxation for Robust alignment
from ncpol2sdpa import generate_variables, SdpRelaxation
from liegroups import SO3
from helpers import *


def make_quassosar_equalities(X, redundancies=None):
    N = int(len(X)/4 - 1)
    equalities = [np.dot(X[0:4], np.transpose(X[0:4])) - 1]
    for i in range(1, N+1):
        for j in range(0, 4):
            for k in range(0, 4):
                equalities.append(X[j]*X[k] - X[i*4+j]*X[i*4+k])

    if redundancies == 'linear':
        for i in range(1, N):
            for k in range(0, 4):
                for l in range(0, 4):
                    equalities.append(X[i*4+k]*X[i*4+l] - X[(i+1)*4+k]*X[(i+1)*4+l])
        for k in range(0, 4):
            for l in range(0, 4):
                equalities.append(X[4+k]*X[4+l] - X[N*4+k]*X[N*4+l])
    elif redundancies == 'full':
        # This ruins the sparsity, may not need it (hopefully)
        for i in range(1, N+1):
            for j in range(1, i):
                for k in range(0, 4):
                    for l in range(0, 4):
                        equalities.append(X[i*4+k]*X[i*4+l] - X[j*4+k]*X[j*4+l])
    return equalities


def make_quassosar_objective(X, q1, q2, c_bar_2=1, sigma_2_i=1):
    Q = build_cost_function_matrix(q1, q2, c_bar_2, sigma_2_i)
    return np.dot(X, np.dot(Q, np.transpose(X))), Q


def make_quassosar_sdp(q1, q2, c_bar_2, level=1, redundancies=None, sparsity=True):
    N = q1.shape[0]
    X = generate_variables('x', 4*(N+1))
    equalities = make_quassosar_equalities(X, redundancies=redundancies)
    obj, Q = make_quassosar_objective(X, q1, q2, c_bar_2=c_bar_2)
    sdp = SdpRelaxation(X)
    if sparsity:
        sdp.variables = custom_sparsity(sdp.variables, N, redundancies=redundancies)
    # The chordal_extension=True option does not work, so we use our custom sparsity
    sdp.get_relaxation(level, objective=obj, equalities=equalities)#, chordal_extension=True)
    return sdp, X, Q


def extract_sparse_pop_solution(sdp, X):
    q = np.zeros(4)
    q[0] = np.sqrt(sdp[X[0] ** 2])
    q[1] = np.sqrt(sdp[X[1] ** 2])
    q[2] = np.sqrt(sdp[X[2] ** 2])
    q[3] = np.sqrt(sdp[X[3] ** 2])
    if sdp[X[0]*X[3]] < 0.:
        q[0] *= -1.
    if sdp[X[1]*X[3]] < 0.:
        q[1] *= -1.
    if sdp[X[2]*X[3]] < 0.:
        q[2] *= -1.

    N = int(len(X)/4 - 1)
    outlier_list = []
    for idx in range(1, N+1):
        if sdp[X[0]*X[4*idx]] < 0:
            outlier_list.append(idx-1)
    return q, outlier_list


def custom_sparsity(variables, N, redundancies=None):
    if not redundancies:
        return [variables[0:4] + variables[4 * i:4 * (i + 1)] for i in range(1, N + 1)]
    elif redundancies == 'linear':
        # Create cliques for redundant constraints
        cliques = [variables[0:4] + variables[4 * i:4 * (i + 1)] + variables[4 * (i + 1):4 * (i + 2)] for i in range(1, N)]
        cliques.append(variables[0:4] + variables[4:8] + variables[-4:])
        return cliques
    elif redundancies == 'full':
        cliques = [variables[0:4] + variables[4 * i:4 * (i + 1)] + variables[4 * j:4 * (j+1)] for i in range(1, N) for j in range(1, i)]
        cliques.append(variables[0:4] + variables[4:8] + variables[-4:])
        return cliques


def solve_quassosar(q1, q2, c_bar_2, level=1, redundancies=None, sparsity=True):
    sdp, X, Q = make_quassosar_sdp(q1, q2, c_bar_2, level=level, sparsity=sparsity, redundancies=redundancies)
    sdp.solve(solver='mosek')
    # gap = sdp.primal-sdp.dual
    t_solve = sdp.solution_time
    q_est, outlier_inds = extract_sparse_pop_solution(sdp, X)
    N = q1.shape[0]
    q_full_est = np.zeros((4 * (N + 1), 1))
    q_full_est[0:4, 0] = q_est
    for idx in range(1, N + 1):
        if idx - 1 in outlier_inds:
            q_full_est[4 * idx:4 * (idx + 1), 0] = -q_est
        else:
            q_full_est[4 * idx:4 * (idx + 1), 0] = q_est
    gap = np.dot(q_full_est.T, np.dot(Q, q_full_est)) - sdp.dual
    return q_est, outlier_inds, t_solve, gap, sdp.dual


if __name__=='__main__':
    np.random.seed(8675309)
    level = 1
    N = 10
    N_out = 5
    sigma = 0.01
    c_bar_2 = (3*sigma+1e-4)**2
    redundancies =  None  # None #'full'
    q1, q2, C_true = make_random_instance(N, N_out, sigma=sigma)
    q_est, outlier_inds, t_solve, gap, dual = solve_quassosar(q1, q2, c_bar_2, level=1, redundancies=redundancies)
    print('Gap: {:}'.format(gap))
    print('Runtime: {:}'.format(t_solve))
    # Extract solution
    C_est = SO3.from_quaternion(q_est, ordering='xyzw').as_matrix()
    print('Frob. norm error: {:.5f}'.format(np.linalg.norm(C_est - C_true)))
