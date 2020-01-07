import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
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
    path = './saved_data/kitti/diff_lr_kitti_experiment_3models_seq_00_12-19-2019-19-28-32.pt'
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
    output_file = 'plots/' + path.replace('.pt','').replace('saved_data/kitti/','') + '_plot.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


def scatter_shapenet_example():
    pointnet_data = '/Users/valentinp/Dropbox/Postdoc/projects/misc/RotationContinuity/shapenet/data/pc_plane'
    valid_loader = DataLoader(PointNetDataset(pointnet_data + '/points_test', load_into_memory=True, device=torch.device('cpu'), rotations_per_batch=10, dtype=torch.float, test_mode=True),
                        batch_size=1, pin_memory=True, collate_fn=pointnet_collate,
                        shuffle=True, num_workers=1, drop_last=False)
    
    N = 4
    fig = plt.figure()
    fig.set_size_inches(4,4)
    
    for i, (x, target) in enumerate(valid_loader):
        pc1 = x[:,0,:,:].transpose(1,2)
        pc2 = x[:,1,:,:].transpose(1,2)

        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.scatter(pc1[0,0,:],pc1[0,1,:],pc1[0,2,:], c='tab:blue',s=0.1, marker=",")
        ax.scatter(pc2[0,0,:],pc2[0,1,:],pc2[0,2,:], c='tab:green',s=0.1, marker=",")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.axis('off')
        ax.autoscale_view('tight')
        
        if i == N - 1:
            break

    output_file = 'plots/shapenet_plain.pdf'
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

def rotmat_angle_table_stats():
    path = 'saved_data/synthetic/rotangle_synthetic_wahba_experiment_3models_dynamic_01-06-2020-19-35-48.pt'
    data = torch.load(path)
    args = data['args']
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float
    
    fig, axes = plt.subplots(ncols=3, sharey=True)
    fig.subplots_adjust(wspace=0)
    fig.set_size_inches(4,2)

    for m_i, max_angle in enumerate(data['max_angles']):
        train_data, test_data = create_experimental_data_fast(args.N_train, args.N_test, args.matches_per_sample, max_rotation_angle=max_angle, sigma=args.sim_sigma, beachball=False, device=device, dtype=tensor_type)

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


        axes[m_i].boxplot([error_quat, error_6D, error_A])
        axes[m_i].set(xticklabels=['Quat', '6D', 'A (sym)'], xlabel=str(max_angle))
        axes[m_i].margins(0.05) # Optional

        #plt.show()

        #print('{:.2F},{:.2F},{:.2F},{:.2F},{:.2F},{:.2F},{:.2F},{:.2F},{:.2F}'.format(error_quat.min(), error_quat.median(), error_quat.max(), error_6D.min(), error_6D.median(), error_6D.max(), error_A.min(), error_A.median(), error_A.max()))
    
    output_file = 'plots/synthetic_rotangle_box.pdf'
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    #plot_wahba_training_comparisons()
    #plot_learning_rate_wahba_experiment()
    #scatter_shapenet_example()
    rotmat_angle_table_stats()