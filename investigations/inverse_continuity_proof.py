import sympy as sym
import numpy as np


def omega_left(q):

    M = sym.Matrix([[q[3], -q[2], q[1], q[0]],
                    [q[2], q[3], -q[0], q[1]],
                    [-q[1], q[0], q[3], q[2]],
                    [-q[0], -q[1], -q[2], q[3]]])

    return M


def omega_right(q):
    M = sym.Matrix([[q[3], q[2], -q[1], q[0]],
                    [-q[2], q[3], q[0], q[1]],
                    [q[1], -q[0], q[3], q[2]],
                    [-q[0], -q[1], -q[2], q[3]]])

    return M

if __name__ == '__main__':

    q = sym.symbols('q0:4')
    q = sym.Matrix(q)
    q_inv = -q
    q_inv[-1] = -q_inv[-1]
    ## Householder check
    # w = sym.Matrix(q)
    # w[0] = w[0] + sym.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    #
    # w_neg = -sym.Matrix(q)
    # w_neg[0] = w_neg[0] + sym.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    #
    # Q = sym.eye(4) - 2*(w*w.T)/(w.dot(w))
    # Q[0, :] = sym.zeros(1, 4)
    # A = Q.T*Q
    # Q_neg = sym.eye(4) - 2*(w_neg*w_neg.T)/(w_neg.dot(w_neg))
    # Q_neg[0, :] = sym.zeros(1, 4)
    # A_neg = Q_neg.T*Q_neg
    #
    # print("Are they equal?!? Should be zeros:")
    # print(sym.simplify(A - A_neg))

    ##
    # c1 = sym.symbols('c0:3')
    # c1_hat = sym.zeros(4, 1)
    # c1_hat[0:3, 0] = c1
    e1_hat = sym.zeros(4, 1)
    e1_hat[0] = 1
    c1_hat = omega_left(q)*omega_right(q_inv)*e1_hat

    M_l = omega_left(c1_hat)
    M_r = omega_right(e1_hat)

    # c2 = sym.symbols('d0:3')
    # c2_hat = sym.zeros(4, 1)
    # c2_hat[0:3, 0] = c2
    e2_hat = sym.zeros(4, 1)
    e2_hat[1] = 1
    c2_hat = omega_left(q) * omega_right(q_inv) * e2_hat

    M_l2 = omega_left(c2_hat)
    M_r2 = omega_right(e2_hat)

    A1 = sym.eye(4) + M_l * M_r
    A2 = sym.eye(4) + M_l2*M_r2

