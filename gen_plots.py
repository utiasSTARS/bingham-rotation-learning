import numpy as np


def _plot_curve(ax, x, y, label, color):
    ax.plot(x, y,  c=color, linewidth=0.75, label=label)
    return

def plot_training_plots(filename):
    check_point = torch.load(scene_checkpoint, map_location=lambda storage, loc: storage)
    (q_gt, q_est, R_est, R_direct_est) = (check_point['predict_history'][0],
                                            check_point['predict_history'][1],
                                            check_point['predict_history'][2],
                                            check_point['predict_history'][3])
    fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')

    R_est = R_est.numpy()
    R_direct_est = R_direct_est.numpy()

    _plot_curve(x_labels, phi_est[:, 0], phi_gt[:, 0], np.sqrt(R_est[:,0,0].flatten()) * deg_factor,  '$\phi_1$ (deg)', ax[0])
    _plot_curve(x_labels, phi_est[:, 1], phi_gt[:, 1], np.sqrt(R_est[:,1,1].flatten()) * deg_factor, '$\phi_2$ (deg)', ax[1])

    ax[2].legend()
    ax[2].set_xlabel('Frame')
    output_file = '7scenes_abs_' + scene_checkpoint.replace('.pt','').replace('7scenes_data/','') + '.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    
    if y_lim is not None:
        ax.set_ylim(y_lim)
