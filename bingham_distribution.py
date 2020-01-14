import numpy as np
from scipy.integrate import tplquad
from scipy.interpolate import Rbf
import functools
from numpy import sin, cos

import torch


def bingham_integrand(phi3, phi2, phi1, lambdas):
    """Lambdas should be an array of 4 elements (4th is zero if going by convention) """
    t = np.array([sin(phi1)*sin(phi2)*sin(phi3),
                  sin(phi1)*sin(phi2)*cos(phi3),
                  sin(phi1)*cos(phi2),
                  cos(phi1)])
    exponent = np.sum(lambdas*(t**2))
    return np.exp(exponent)*sin(phi1)**2*sin(phi2)


def create_bingham_interpolator(data_file):
    pass


def bingham_normalization(lambdas, ):
    """Compute the Bingham normalization of concentration parameters.
    """
    assert len(lambdas) == 4
    f_integrand = functools.partial(bingham_integrand, lambdas=lambdas)

    return tplquad(f_integrand, 0., np.pi,
                   lambda a: 0., lambda a: np.pi,
                   lambda a, b: 0., lambda a, b: 2.*np.pi,
                   epsabs=1e-7, epsrel=1e-3)


def bingham_dist(q, lambdas, coeff_N=None):

    if coeff_N == None:
        coeff_N, _ = bingham_normalization(lambdas)

    return np.exp(np.sum(lambdas*(q**2)))/coeff_N


class RadialBasisFunction(torch.nn.Module):
    """
    Differentiable Radial Basis Function Layer
    Input:
    Output:
    """

    def __init__(self, rbf_model):
        super(RadialBasisFunction, self).__init__()
        self.rbf_model = rbf_model

    @staticmethod
    def forward(ctx, A_vec):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


if __name__ == '__main__':

    lambdas = np.linspace(0.0, 3.0, 4)
    coeff_N, err_est = bingham_normalization(lambdas)
    print('Bingham normalization coefficient: {:}'.format(coeff_N))
    print('Bingham normalization coefficient error est: {:}'.format(err_est))

    lambdas_shifted = lambdas + 10.0
    coeff_N_shifted, err_est_shifted = bingham_normalization(lambdas_shifted)
    print('Bingham normalization coefficient: {:}'.format(coeff_N_shifted))
    print('Bingham normalization coefficient error est: {:}'.format(err_est_shifted))


    # Check invariance
    q = np.random.rand(4)
    q = q/np.linalg.norm(q)
    p1 = bingham_dist(q, lambdas, coeff_N=coeff_N)
    p2 = bingham_dist(q, lambdas_shifted, coeff_N=coeff_N_shifted)

    print('Bingham PDF likelihood: {:}'.format(p1))
    print('Bingham PDF shifted likelihood: {:}'.format(p2))

