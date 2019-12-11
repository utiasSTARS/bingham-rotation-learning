import numpy as np
import cvxpy as cp

def rotation_matrix_constraints(redundant=True):
    R = cp.variable((3, 3))
    row_constraints = cp.matmul(R, R.T) - np.eye(3)
    if redundant:
        col_constraints = cp.matmul(R.T, R) - np.eye(3)

    # Re-arrange into SDP-relaxable formulation

if __name__=='__main__':

    n = 10