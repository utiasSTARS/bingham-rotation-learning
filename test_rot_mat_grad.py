import numpy as np
import torch


if __name__ == '__main__':
    b = 5
    m = 2
    n = 3

    E = np.zeros((m, n, n))
    E[0, :, :] = np.array([[1., 2., 3.],
                           [4., 5., 6.],
                           [7., 8., 9.]])
    E[1, :, :] = 3*E[0, :, :]
    mu = np.array([[0., 1.],
                   [2., 3.],
                   [4., 5.],
                   [6., 7.],
                   [8., 9.]])
    x = np.array([[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.],
                  [9., 10., 11.],
                  [12., 13., 14.]])

    E = torch.from_numpy(E)
    mu = torch.from_numpy(mu)
    x = torch.from_numpy(x)
    print(E)
    print(mu)
    print(x)


    print("E and mu: ")
    # M1 = E[None, :, :, :] * mu[:, :, None, None]
    # print(M1)
    M2 = torch.einsum('bi,imn->bmn', mu, E)
    print(M2)
    print("E and x: ")
    B = torch.einsum('mij,bj->bim', E, x)
    print(B)

