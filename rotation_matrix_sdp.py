import numpy as np
import cvxpy as cp

def rotation_matrix_constraints(redundant=True, right_handed=True, homogeneous=True):
    '''
    Return QCQP/SDP constraint matrices enforcing rotation matrices.

    :param redundant: indicates whether to include redundant column orthogonality constraints
    :param right_handed: indicates whether to enforce SO(3) right-handedness
    :param homogeneous: indicates whether to include a homogenizing variably y^2 = 1
    :return: array of quadratic constraint matrices, vector of constraint constants
    '''
    if right_handed:
        homogeneous = True

    if homogeneous:
        N = 10
    else:
        N = 9
    constraint_matrices = np.zeros((0, N, N))
    c = []

    # Homogeneous constraint
    if homogeneous:
        A_h = np.zeros((N, N))
        A_h[-1, -1] = 1
        constraint_matrices = np.append(constraint_matrices,
                                        np.expand_dims(A_h, axis=0), axis=0)
        c.append(1)

    # Row constraints
    for idx in range(3):
        ind1 = slice(3 * idx, 3 * (idx + 1))
        for jdx in range(idx+1):
            ind2 = slice(3 * jdx, 3 * (jdx + 1))
            A = np.zeros((N, N))
            if idx == jdx:
                A[ind1, ind1] = np.eye(3)
                c.append(1)
            else:
                A[ind1, ind2] = 0.5*np.eye(3)
                A[ind2, ind1] = 0.5*np.eye(3)
                c.append(0)
            constraint_matrices = np.append(constraint_matrices,
                                        np.expand_dims(A, axis=0), axis=0)

    # Column constraints
    if redundant:
        for idx in range(3):
            for jdx in range(idx+1):
                A = np.zeros((N, N))
                if idx == jdx:
                    A_sub = np.zeros((3, 3))
                    A_sub[idx, idx] = 1
                    A[0:3, 0:3] = A_sub
                    A[3:6, 3:6] = A_sub
                    A[6:9, 6:9] = A_sub
                    c.append(1)
                else:
                    A_sub = np.zeros((3, 3))
                    A_sub[idx, jdx] = 0.5
                    A_sub[jdx, idx] = 0.5
                    A[0:3, 0:3] = A_sub
                    A[3:6, 3:6] = A_sub
                    A[6:9, 6:9] = A_sub
                    c.append(0)
                constraint_matrices = np.append(constraint_matrices,
                                        np.expand_dims(A, axis=0), axis=0)

    # Right-handed constraint
    if right_handed:
        A_sub = -1*np.array([[[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                          [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                          [[0, -1, 0], [1, 0, 0], [0, 0, 0]]])

        for idx in range(0, 3):
            # Cyclic constraints {1,2,3}, {2,3,1}, {3,1,2}
            ind1 = np.arange(3*idx, 3*(idx+1))
            ind2 = slice(3*((idx+1)%3), 3*((idx+1)%3 + 1))
            ind3 = np.arange(3*((idx+2) % 3), 3*((idx+2) % 3 + 1))
            for jdx in range(0, 3):
                A = np.zeros((N, N))
                A[ind1, ind2] = A_sub[jdx, :, :]
                A[ind3[jdx], -1] = -1
                A = 0.5*(A+A.T)
                constraint_matrices = np.append(constraint_matrices,
                                        np.expand_dims(A, axis=0), axis=0)
                c.append(0)

    return constraint_matrices, np.array(c)

def solve_rotation_SDP(cost_matrix):

    Z = cp.Variable((10, 10), PSD=True)

if __name__=='__main__':

    n = 10

    constraint_matrices, c_vec = rotation_matrix_constraints()
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r = np.reshape(R.T, (9, -1))
    r = np.vstack((r, 1))

    R2 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    r2 = np.reshape(R2.T, (9, -1))
    r2 = np.vstack((r2, 1))

    print('Valid SO(3):')
    for idx in range(constraint_matrices.shape[0]):
        print(np.dot(r.T, np.dot(constraint_matrices[idx, :, :], r)) - c_vec[idx])

    print('Valid O(3):')
    for idx in range(constraint_matrices.shape[0]):
        print(np.dot(r2.T, np.dot(constraint_matrices[idx, :, :], r2)) - c_vec[idx])