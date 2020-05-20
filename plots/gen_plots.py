import numpy as np
import torch
import matplotlib
from matplotlib.colors import to_rgba
import math
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'..')
from loaders import PointNetDataset, pointnet_collate
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from networks import *
from helpers_sim import *

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
    datafile = '../saved_data/synthetic/synthetic_wahba_experiment_12-06-2019-01-20-24.pt'
    data = torch.load(datafile, map_location=lambda storage, loc: storage)

    train_stats_direct = data['train_stats_direct'].detach().numpy()
    train_stats_rep = data['train_stats_rep'].detach().numpy()
    test_stats_direct = data['test_stats_direct'].detach().numpy()
    test_stats_rep = data['test_stats_rep'].detach().numpy()

    #Individual plots
    if individual:
        fig = _create_training_fig(train_stats_direct, train_stats_rep)
        output_file = 'plots/' + datafile.replace('.pt','').replace('../saved_data/synthetic/','') + '_train.pdf'
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)

        fig = _create_training_fig(test_stats_direct, test_stats_rep)
        output_file = 'plots/' + datafile.replace('.pt','').replace('../saved_data/synthetic/','') + '_test.pdf'
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
    
    if combined:
        fig = _create_training_fig_combined(train_stats_direct, train_stats_rep, test_stats_direct, test_stats_rep)
        output_file = 'plots/' + datafile.replace('.pt','').replace('../saved_data/synthetic/','') + '_combined.pdf'
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)

def _plot_curve_with_bounds(ax, x, y, lower, upper, label, color):
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.fill_between(x, lower, upper, alpha=0.3, facecolor=color)
    ax.plot(x, y, color, linewidth=1.5, label=label)
    return 

def _create_learning_rate_fig_combined(args, train_err, test_err, names, legend=True):
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
    fig.set_size_inches(4, 1.5)

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
    
    if legend:   
        ax[0].legend(fontsize='small', ncol=2, columnspacing=0.4, handletextpad=0.4, labelspacing=0.3, borderpad=0.25, loc='upper right', handlelength=1.0)

    ax[0].set_xlabel('epoch (\\textit{train})')
    ax[1].set_xlabel('epoch (\\textit{test})')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_ylabel('mean error (deg)')
    return fig

def plot_learning_rate_wahba_experiment():
    path = '../saved_data/synthetic/diff_lr_synthetic_wahba_experiment_3models_chordal_dynamic_01-16-2020-04-24-48.pt'
    custom_legend = ['\\texttt{6D}','\\texttt{quat}','$\mathbf{A}$ (\\textit{ours})']
    plot_learning_rate_experiment(path, custom_legend)

def plot_learning_rate_shapenet_experiment():
    #path = '../saved_data/shapenet/diff_lr_shapenet_experiment_3models_01-24-2020-03-03-36.pt'
    path = '../saved_data/shapenet/diff_lr_shapenet_experiment_3models_01-25-2020-00-56-49.pt'
    custom_legend = ['\\texttt{6D}','\\texttt{quat}','$\mathbf{A}$ (\\textit{ours})']
    plot_learning_rate_experiment(path, custom_legend)

def plot_learning_rate_experiment(data_path, custom_legend=None):
    checkpoint = torch.load(data_path)
    args = checkpoint['args']
    print(args)
    train_stats_list = checkpoint['train_stats_list']
    test_stats_list = checkpoint['test_stats_list']
    if custom_legend is not None:
        names = custom_legend
    else:
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
    output_file =  data_path.split('/')[-1].replace('.pt','') + '_plot.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


def scatter_shapenet_example():
    pointnet_data = '/Users/valentinp/Dropbox/Postdoc/projects/misc/RotationContinuity/shapenet/data/pc_plane'
    valid_loader = DataLoader(PointNetDataset(pointnet_data + '/points_test', load_into_memory=True, device=torch.device('cpu'), rotations_per_batch=10, dtype=torch.float, test_mode=True),
                        batch_size=1, pin_memory=True, collate_fn=pointnet_collate,
                        shuffle=False, num_workers=1, drop_last=False)
    

    N = 4
    fig = plt.figure()
    fig.set_size_inches(8,2)
    
    for i, (x, _) in enumerate(valid_loader):
        pc1 = x[:,0,:,:].transpose(1,2)
        pc2 = x[:,1,:,:].transpose(1,2)

        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        ax.scatter(pc1[0,0,:],pc1[0,1,:],pc1[0,2,:], c='tab:grey',s=0.001, marker=",")
        ax.scatter(pc2[0,0,:],pc2[0,1,:],pc2[0,2,:], c='tab:green',s=0.001, marker=",", alpha=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.autoscale_view('tight')
        ax.axis('off')
        #ax.set_alpha(0.1)

        if i == N - 1:
            break

    output_file = 'shapenet/shapenet_vis_{}_clouds.pdf'.format(N)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


def test_wabha_model(model, x, targets, **kwargs):
    #model.eval() speeds things up because it turns off gradient computation
    model.eval()
    # Forward
    with torch.no_grad():
        out = model.forward(x, **kwargs)
        loss = loss_fn(out, targets)
    return (out, loss.item())

def rotmat_angle_table_stats(cache_data=True):
    
    if cache_data:
        path = '../saved_data/synthetic/rotangle_synthetic_wahba_experiment_3models_dynamic_01-06-2020-19-35-48.pt'
        data = torch.load(path)
        args = data['args']
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 
        tensor_type = torch.double if args.double else torch.float
        
        plot_angles = [10, 100, 180]
        a_i = 0
        maxrot_data = []
        for m_i, max_angle in enumerate(data['max_angles']):
            if max_angle not in plot_angles:
                continue
            _, test_data = create_experimental_data_fast(args.N_train, 500, args.matches_per_sample, max_rotation_angle=max_angle, sigma=args.sim_sigma, beachball=False, device=device, dtype=tensor_type)

            model_6D = RotMat6DDirect().to(device=device, dtype=tensor_type)
            model_quat = PointNet(dim_out=4, normalize_output=True).to(device=device, dtype=tensor_type)
            model_A_sym = QuatNet(enforce_psd=False, unit_frob_norm=True).to(device=device, dtype=tensor_type)

            model_6D.load_state_dict(data['models_6D'][m_i], strict=False)        
            model_quat.load_state_dict(data['models_quat'][m_i], strict=False)        
            model_A_sym.load_state_dict(data['models_A_sym'][m_i], strict=False)


            with torch.no_grad():
                q_quat = model_quat.forward(test_data.x).cpu()
                q_A = model_A_sym.forward(test_data.x).cpu()
                q_6D = rotmat_to_quat(model_6D.forward(test_data.x).cpu())
            
            error_quat = quat_angle_diff(q_quat, test_data.q.cpu(), reduce=False).squeeze().numpy()
            error_A = quat_angle_diff(q_A, test_data.q.cpu(), reduce=False).squeeze().numpy()
            error_6D = quat_angle_diff(q_6D, test_data.q.cpu(), reduce=False).squeeze().numpy()

            #print('Max Angle: {:.2F}'.format(max_angle))
            # print('Quat | Min {:.2F}, Median {:.2F}, Max {:.2F},'.format(error_quat.min(), error_quat.median(), error_quat.max()))
            # print('6D | Min {:.2F}, Median {:.2F}, Max {:.2F},'.format(error_6D.min(), error_6D.median(), error_6D.max()))
            # print('A_sym | Min {:.2F}, Median {:.2F}, Max {:.2F},'.format(error_A.min(), error_A.median(), error_A.max()))
            
            #print('Quat | 6D | A (sym)')

            maxrot_data.append((error_quat, error_6D, error_A))

        desc = path.split('/')[-1].split('.pt')[0]
        processed_data_file = '../saved_data/synthetic/'+'processed_{}.pt'.format(desc)
        
        torch.save({
                    'maxrot_data': maxrot_data,
                    'plot_angles': plot_angles
        }, processed_data_file)
        print('Saved data to {}.'.format(processed_data_file))
    else:
        processed_data_file = '../saved_data/synthetic/processed_rotangle_synthetic_wahba_experiment_3models_dynamic_01-06-2020-19-35-48.pt'

    processed_data = torch.load(processed_data_file)
    fig, axes = plt.subplots(ncols=3, sharey=True)
    fig.subplots_adjust(wspace=0)
    fig.set_size_inches(4,1.5)

    lw = 0.25
    colours = ['tab:green', 'tab:red', 'tab:blue']
    
    boxprops = dict(linewidth=0.4) 
    whiskerprops = dict(linewidth=2*lw)
    flierprops = dict(marker='o', markersize=5,
                  markeredgewidth=lw)
    medianprops = dict(linewidth=lw, color='k')


    for a_i in range(len(processed_data['maxrot_data'])):
        max_angle = processed_data['plot_angles'][a_i]
        error_quat, error_6D, error_A = processed_data['maxrot_data'][a_i]
        bp = axes[a_i].boxplot([np.log10(error_quat), np.log10(error_6D), np.log10(error_A)], widths=0.6, patch_artist=True, flierprops=flierprops, notch=True, boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops)
        axes[a_i].set(xticklabels=['\\texttt{quat}', '\\texttt{6D}', '$\mathbf{A}$'], xlabel=str(max_angle)+'°')
        
        for p_i in range(len(bp['boxes'])):
            bp['boxes'][p_i].set(facecolor=to_rgba(colours[p_i], alpha=0.4), edgecolor=colours[p_i])
            bp['medians'][p_i].set(color=colours[p_i])
            bp['fliers'][p_i].set(markeredgecolor=colours[p_i])

        for p_i in range(len(bp['whiskers'])):
            c_i = math.floor(p_i/2)
            bp['whiskers'][p_i].set(color=colours[c_i])
            bp['caps'][p_i].set(color=colours[c_i])

                 

        # axes[a_i].set_yscale('log')
        axes[a_i].set_yticks([-1, 0, 1, 2], minor=True)
        axes[a_i].grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.25)

        if a_i == 0:
            axes[a_i].set_ylabel('$\log_{10}$ error (°)')
    

        
    desc = processed_data_file.split('/')[3].split('.pt')[0]
    output_file = 'maxrotangle_{}.pdf'.format(desc)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    #plot_wahba_training_comparisons()
    #plot_learning_rate_wahba_experiment()
    #plot_learning_rate_shapenet_experiment()
    #scatter_shapenet_example()
    rotmat_angle_table_stats(cache_data=False)