import numpy as np
from matplotlib import pyplot as plt

from helpers_sim import build_A, gen_sim_data
# from gen_plots import _plot_curve_with_bounds # Required cv2


def _plot_curve_with_bounds(ax, x, y, lower, upper, label, color):
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.fill_between(x, lower, upper, alpha=0.3, facecolor=color)
    ax.plot(x, y, color, linewidth=1.5, label=label)
    return


def _gen_eigenvalue_gap_plot(sigma, gap_data):
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
    output_file = 'plots/eigenvalue_gap_vs_noise_non_normalized.pdf'
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel('eigenvalue gap')
    ax.set_xlabel('std. deviation $\sigma$ (m)')
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')
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

    _gen_eigenvalue_gap_plot(sigma_vec, gap_data)

np.linalg.solve