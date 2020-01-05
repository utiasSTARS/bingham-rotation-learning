import numpy as np
from matplotlib import pyplot as plt


def wigner_surmise(s):
    return 0.5*np.pi*s*np.exp(-0.25*np.pi*s**2)


if __name__ == '__main__':
    # Parameters
    n = 4
    N = int((n+1)*n/2)
    n_runs = 100000
    sigma = 1.0
    mu = 0.0
    inds = np.triu_indices(n)
    diag_inds = np.diag_indices(n)
    spacings = np.zeros((n_runs, n-1))
    # Create random Gaussian matrices
    for idx in range(n_runs):
        A = np.random.randn(n, n)
        A[inds[1], inds[0]] = A[inds[0], inds[1]]
        A[diag_inds] = np.random.rand(n)*2.0 - 1.0
        eigenvals = np.linalg.eigvalsh(A)
        spacings[idx, :] = np.diff(eigenvals)

    spacings = np.reshape(spacings, (1, -1)).squeeze()
    mean_gap = np.mean(spacings)

    # Wigner Surmise PDF curve
    s = np.linspace(0.0, 5.0, 10000)
    p = wigner_surmise(s)

    plt.figure()
    plt.hist(spacings/mean_gap, color='green', bins=100, density=True, histtype='stepfilled')
    plt.grid()
    plt.plot(s, p, 'k-')
    plt.xlabel('Eigenvalue Gap/Mean Eigenvalue Gap')
    plt.ylabel('Probability')
    plt.title('Wigner Surmise (Black) Over Mixed Uniform Diagonal and Gaussian Off-Diagonal Samples')
    plt.show()
