import numpy as np
from matplotlib import pyplot as plt
from utils import normalized
from helpers_sim import build_A, gen_sim_data
# from gen_plots import _plot_curve_with_bounds # Required cv2


def _plot_curve_with_bounds(ax, x, y, lower, upper, label, color):
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.fill_between(x, lower, upper, alpha=0.3, facecolor=color)
    ax.plot(x, y, color, linewidth=1.5, label=label)
    return


def _gen_eigenvalue_gap_plot(sigma, gap_data, save_file, xlabel_str, ylabel_str):
    plt.rc('text', usetex=True)
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches(4, 2)
    colours = ['tab:red', 'tab:green', 'tab:blue', 'tab:grey']
    for idx in range(3):
        _plot_curve_with_bounds(ax, sigma, np.mean(gap_data[idx, :, :], axis=1),
                                np.quantile(gap_data[idx, :, :], 0.1, axis=1),
                                np.quantile(gap_data[idx, :, :], 0.9, axis=1),
                                '$\lambda_' + str(idx+2) + '- \lambda_1$',
                                colours[idx])

    ax.legend()
    # ax.set_yscale('log')
    ax.set_ylabel(ylabel_str)
    ax.set_xlabel(xlabel_str)
    fig.tight_layout()
    fig.savefig(save_file, bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':

    N_runs = 1000
    N = 50

    sigma_vec = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    gap_data = np.zeros((3, len(sigma_vec), N_runs))
    for idx in range(len(sigma_vec)):
        sigma = sigma_vec[idx]*np.ones(N)
        for jdx in range(N_runs):
            _, x_1, x_2 = gen_sim_data(N, sigma)
            A = build_A(x_1, x_2, sigma**2)
            # A = A/np.linalg.norm(A, ord='fro')
            eigvalues = np.linalg.eigvalsh(A)
            gap_data[0, idx, jdx] = eigvalues[1] - eigvalues[0]
            gap_data[1, idx, jdx] = eigvalues[2] - eigvalues[0]
            gap_data[2, idx, jdx] = eigvalues[3] - eigvalues[0]

    _gen_eigenvalue_gap_plot(sigma_vec, gap_data, '../plots/eigenvalue_gap_vs_noise.pdf',
                             'std. deviation $\sigma$ (m)',
                             'eigenvalue gap')


    outlier_rate_vec = np.linspace(0.0, 0.7, 8)
    gap_data_outlier = np.zeros((3, len(sigma_vec), N_runs))
    for idx in range(len(outlier_rate_vec)):
        sigma = 0.01*np.ones(N)
        n_outliers = int(N*outlier_rate_vec[idx])
        for jdx in range(N_runs):
            _, x_1, x_2 = gen_sim_data(N, sigma)
            perm = np.random.permutation(N)
            outlier_inds = perm[0:n_outliers]
            x_2[outlier_inds, :] = np.random.rand(n_outliers, 3)
            x_2[outlier_inds, :] = normalized(x_2[outlier_inds, :], axis=1)
            A = build_A(x_1, x_2, sigma**2)
            # A = A/np.linalg.norm(A, ord='fro')
            eigvalues = np.linalg.eigvalsh(A)
            gap_data_outlier[0, idx, jdx] = eigvalues[1] - eigvalues[0]
            gap_data_outlier[1, idx, jdx] = eigvalues[2] - eigvalues[0]
            gap_data_outlier[2, idx, jdx] = eigvalues[3] - eigvalues[0]

    _gen_eigenvalue_gap_plot(outlier_rate_vec*100, gap_data_outlier, '../plots/eigenvalue_gap_vs_outlier_rate.pdf',
                             'outlier rate (\%)',
                             'eigenvalue gap')
