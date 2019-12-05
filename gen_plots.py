import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _plot_curve(ax, x, y, label, color):
    ax.grid(True, which='both')
    ax.plot(x, y,  c=color, linewidth=1.5, label=label)
    return

def _create_training_fig(stats_direct, stats_rep):
    epochs = np.arange(stats_direct.shape[0])
    fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')
    _plot_curve(ax[0], epochs, stats_direct[:, 0],  'direct', 'r')
    _plot_curve(ax[0], epochs, stats_rep[:, 0],  'ours', 'g')

    ax[0].set_ylabel('loss')

    _plot_curve(ax[1], epochs, stats_direct[:, 1],  'direct', 'r')
    _plot_curve(ax[1], epochs, stats_rep[:, 1],  'ours', 'g')
    ax[1].set_ylabel('mean error (deg)')

    ax[0].legend()
    ax[1].set_xlabel('epoch')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    return fig
def plot_wahba_training_comparisons():
    datafile = './saved_data/synthetic/synthetic_wahba_experiment_2019-12-05-17-37-13.pt'
    data = torch.load(datafile, map_location=lambda storage, loc: storage)

    train_stats_direct = data['train_stats_direct'].detach().numpy()
    train_stats_rep = data['train_stats_rep'].detach().numpy()
    fig = _create_training_fig(train_stats_direct, train_stats_rep)
    output_file = 'plots/' + datafile.replace('.pt','').replace('saved_data/synthetic/','') + '_train.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

    test_stats_direct = data['test_stats_direct'].detach().numpy()
    test_stats_rep = data['test_stats_rep'].detach().numpy()
    fig = _create_training_fig(test_stats_direct, test_stats_rep)
    output_file = 'plots/' + datafile.replace('.pt','').replace('saved_data/synthetic/','') + '_test.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    
    


if __name__=='__main__':
    plot_wahba_training_comparisons()