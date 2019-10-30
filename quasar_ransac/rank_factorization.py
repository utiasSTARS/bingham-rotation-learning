import numpy as np
from pymanopt.manifolds import PSDFixedRank


if __name__=='__main__':

    ## Parameters
    N = 10
    N_outliers = 1
    sigma = 0.01
    rank_max = 1
    # Rank 2 real PSD matrices
    manifold = PSDFixedRank(4*(N+1), rank_max)


