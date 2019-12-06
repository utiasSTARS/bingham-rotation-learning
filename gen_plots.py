import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt

def _plot_curve(ax, x, y, label, style):
    ax.grid(True, which='both')
    ax.plot(x, y,  style, linewidth=1.5, label=label)
    return

def _create_training_fig(stats_direct, stats_rep):
    epochs = np.arange(stats_direct.shape[0])
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')
    fig.set_figwidth(2)
    _plot_curve(ax[0], epochs, stats_direct[:, 0],  'direct', 'tab:red')
    _plot_curve(ax[0], epochs, stats_rep[:, 0],  'ours', 'tab:blue')

    ax[0].set_ylabel('loss')

    _plot_curve(ax[1], epochs, stats_direct[:, 1],  'direct', 'tab:red')
    _plot_curve(ax[1], epochs, stats_rep[:, 1],  'ours', 'tab:blue')
    ax[1].set_ylabel('mean error (deg)')

    ax[0].legend()
    ax[1].set_xlabel('epoch')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    return fig

def _create_training_fig_combined(train_stats_direct, train_stats_rep, test_stats_direct, test_stats_rep):
    epochs = np.arange(train_stats_direct.shape[0])
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')
    fig.set_size_inches(4,2)
    _plot_curve(ax[0], epochs, train_stats_direct[:, 0],  'train (direct)', 'r')
    _plot_curve(ax[0], epochs, train_stats_rep[:, 0],  'train (ours)', 'b')
    _plot_curve(ax[0], epochs, test_stats_direct[:, 0],  'validation (direct)', '--r')
    _plot_curve(ax[0], epochs, test_stats_rep[:, 0],  'validation (ours)', '--b')

    ax[0].set_ylabel('loss')

    _plot_curve(ax[1], epochs, train_stats_direct[:, 1],  'train (direct)', 'r')
    _plot_curve(ax[1], epochs, train_stats_rep[:, 1],  'train (ours)', 'b')
    _plot_curve(ax[1], epochs, test_stats_direct[:, 1],  'validation (direct)', '--r')
    _plot_curve(ax[1], epochs, test_stats_rep[:, 1],  'validation (ours)', '--b')

    ax[1].set_ylabel('mean error (deg)')
    ax[0].legend()
    ax[1].set_xlabel('epoch')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')   

    return fig

def plot_wahba_training_comparisons(individual=True, combined=False):
    datafile = './saved_data/synthetic/synthetic_wahba_experiment_12-05-2019-18-24-14.pt'
    data = torch.load(datafile, map_location=lambda storage, loc: storage)

    train_stats_direct = data['train_stats_direct'].detach().numpy()
    train_stats_rep = data['train_stats_rep'].detach().numpy()
    test_stats_direct = data['test_stats_direct'].detach().numpy()
    test_stats_rep = data['test_stats_rep'].detach().numpy()

    #Individual plots
    if individual:
        fig = _create_training_fig(train_stats_direct, train_stats_rep)
        output_file = 'plots/' + datafile.replace('.pt','').replace('saved_data/synthetic/','') + '_train.pdf'
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)

        fig = _create_training_fig(test_stats_direct, test_stats_rep)
        output_file = 'plots/' + datafile.replace('.pt','').replace('saved_data/synthetic/','') + '_test.pdf'
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
    
    if combined:
        fig = _create_training_fig_combined(train_stats_direct, train_stats_rep, test_stats_direct, test_stats_rep)
        output_file = 'plots/' + datafile.replace('.pt','').replace('saved_data/synthetic/','') + '_combined.pdf'
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
    


if __name__=='__main__':
    plot_wahba_training_comparisons()