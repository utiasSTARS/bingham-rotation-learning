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
    datafile = './saved_data/synthetic/synthetic_wahba_experiment_12-06-2019-01-20-24.pt'
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

def _plot_curve_with_bounds(ax, x, y, lower, upper, label, color):
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.fill_between(x, lower, upper, alpha=0.3, facecolor=color)
    ax.plot(x, y, color, linewidth=1.5, label=label)
    return 

def _create_learning_rate_fig_combined(args, train_err, test_err, names):
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
    fig.set_size_inches(4, 2)

    x = np.arange(args.epochs)
    colours = ['tab:red', 'tab:green', 'tab:blue', 'tab:grey']

    for i in range(len(names)):
        _plot_curve_with_bounds(
            ax[0], x, np.quantile(train_err[i], 0.5, axis=0),
            np.quantile(train_err[i], 0.1, axis=0), 
            np.quantile(train_err[i], 0.9, axis=0),  
            names[i], colours[i])
        _plot_curve_with_bounds(
            ax[1], x, np.quantile(test_err[i], 0.5, axis=0),
            np.quantile(test_err[i], 0.1, axis=0), 
            np.quantile(test_err[i], 0.9, axis=0),  
            names[i], colours[i])
       
    ax[0].legend()
    ax[0].set_xlabel('epoch (training)')
    ax[1].set_xlabel('epoch (validation)')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_ylabel('mean error (deg)')
    return fig

def plot_learning_rate_wahba_experiment():
    path = './saved_data/shapenet/diff_lr_shapenet_experiment_3models_12-19-2019-10-56-33.pt'
    checkpoint = torch.load(path)
    args = checkpoint['args']
    print(args)
    train_stats_list = checkpoint['train_stats_list']
    test_stats_list = checkpoint['test_stats_list']
    names = checkpoint['named_approaches']

    trials = args.trials
    train_err = np.empty((len(names), trials, args.epochs))
    test_err = np.empty((len(names), trials, args.epochs))
    
    for t_i in range(trials):
        train_stats = train_stats_list[t_i]
        test_stats = test_stats_list[t_i]
        for app_i in range(len(names)):
            #Index 1 is mean angular error, index 0 is loss
            train_err[app_i, t_i, :] = train_stats[app_i][:, 1].detach().numpy()
            test_err[app_i, t_i, :] = test_stats[app_i][:, 1].detach().numpy()
            
    fig = _create_learning_rate_fig_combined(args, train_err, test_err, names)
    output_file = 'plots/' + path.replace('.pt','').replace('saved_data/shapenet/','') + '_plot.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    #plot_wahba_training_comparisons()
    plot_learning_rate_wahba_experiment()