from helpers import *

from quasar import solve_quasar
from quassosar import solve_quassosar
from liegroups import SO3
import shelve

if __name__=='__main__':
    N_vec = np.array([20, 30, 40, 50])
    outlier_rate_vec = np.array([0.5, 0.8, 0.9])
    N_runs = 5
    # Problem data parameters
    sigma_vec = np.array([0.01, 0.1])
    n_solves_total = len(N_vec) * len(outlier_rate_vec) * len(sigma_vec) * N_runs



    t_sos = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    t_sos_lin = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    t_quasar = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    frob_sos = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    frob_sos_lin = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    frob_quasar = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    gap_sos = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    gap_sos_lin = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    gap_quasar = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    obj_sos = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    obj_sos_lin = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    obj_quasar = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))
    outliers_sos = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs), dtype='bool')
    outliers_sos_lin = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs), dtype='bool')
    outliers_quasar = np.zeros((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs), dtype='bool')

    # Run main loop
    for idx in range(0, len(outlier_rate_vec)):
        outlier_rate = outlier_rate_vec[idx]
        for jdx in range(0, len(N_vec)):
            N = N_vec[jdx]
            N_out = int(N*outlier_rate)
            for kdx in range(0, len(sigma_vec)):
                sigma = sigma_vec[kdx]
                c_bar_2 = (3 * sigma + 1e-4) ** 2  # 3-sigma mistake chance for now
                for ldx in range(0, N_runs):
                    print((idx, jdx, kdx, ldx))
                    print((len(outlier_rate_vec), len(N_vec), len(sigma_vec), N_runs))

                    # Generate data
                    q1, q2, C_true, outlier_inds_true = make_random_instance(N, N_out, sigma=sigma)
                    outlier_inds_true.sort()

                    # Run quasar
                    q_est, est_outlier_indices, t_solve, gap, obj_cost = solve_quasar(q1, q2, c_bar_2,
                                                                                      redundant_constraints=True)
                    t_quasar[idx, jdx, kdx, ldx] = t_solve
                    frob_quasar[idx, jdx, kdx, ldx] = np.linalg.norm(C_true - SO3.from_quaternion(q_est, ordering='xyzw').as_matrix())
                    gap_quasar[idx, jdx, kdx, ldx] = gap
                    obj_quasar[idx, jdx, kdx, ldx] = obj_cost
                    outliers_quasar[idx, jdx, kdx, ldx] = outlier_inds_true == est_outlier_indices
                    # Run quassosar
                    q_est_sos, outlier_inds_sos, t_solve_sos, gap_sos_ijk, obj_cost_sos = solve_quassosar(q1, q2, c_bar_2,
                                                                                                          level=1, redundancies=None)
                    t_sos[idx, jdx, kdx, ldx] = t_solve_sos
                    frob_sos[idx, jdx, kdx, ldx] = np.linalg.norm(C_true - SO3.from_quaternion(q_est_sos, ordering='xyzw').as_matrix())
                    gap_sos[idx, jdx, kdx, ldx] = gap_sos_ijk
                    obj_sos[idx, jdx, kdx, ldx] = obj_cost_sos
                    outliers_sos[idx, jdx, kdx, ldx] = outlier_inds_true == outlier_inds_sos

                    # Run quassosar with 'linear' redundancies
                    q_est_sos_lin, outlier_inds_sos_lin, t_solve_sos_lin, gap_sos_ijk_lin, obj_cost_sos_lin = solve_quassosar(q1, q2, c_bar_2,
                                                                                                          level=1,
                                                                                                          redundancies='linear')
                    t_sos_lin[idx, jdx, kdx, ldx] = t_solve_sos_lin
                    frob_sos_lin[idx, jdx, kdx, ldx] = np.linalg.norm(
                        C_true - SO3.from_quaternion(q_est_sos_lin, ordering='xyzw').as_matrix())
                    gap_sos_lin[idx, jdx, kdx, ldx] = gap_sos_ijk_lin
                    obj_sos_lin[idx, jdx, kdx, ldx] = obj_cost_sos_lin
                    outliers_sos_lin[idx, jdx, kdx, ldx] = outlier_inds_true == outlier_inds_sos_lin

    # Save all workspace data with shelve
    filename = 'quasar_experiment1.out'
    my_shelf = shelve.open(filename, 'n')  # 'n' for new
    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()

    ## For restoring data:
    # my_shelf = shelve.open(filename)
    # for key in my_shelf:
    #     globals()[key] = my_shelf[key]
    # my_shelf.close()
