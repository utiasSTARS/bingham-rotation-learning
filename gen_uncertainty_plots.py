from quaternions import *
from networks import *
from helpers_train_test import *
from liegroups.numpy import SO3
import torch
from datetime import datetime
from convex_layers import *
from torch.utils.data import Dataset, DataLoader
from loaders import KITTIVODatasetPreTransformed
import numpy as np
import torchvision
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt


def evaluate_rotmat_model(loader, model, device, tensor_type):
    q_est = []
    q_target = []
    
    with torch.no_grad():
        model.eval()
        print('Evaluating rotmat model...')
        for _, (x, target) in enumerate(loader):
            #Move all data to appropriate device
            x = x.to(device=device, dtype=tensor_type)
            q = rotmat_to_quat(model.forward(x).squeeze().cpu())
            q_est.append(q)
            q_target.append(target.cpu())
            
    q_est = torch.cat(q_est, dim=0)
    q_target = torch.cat(q_target, dim=0)
    
    return (q_est, q_target)

def evaluate_A_model(loader, model, device, tensor_type):
    q_est = []
    q_target = []
    A_pred = []

    with torch.no_grad():
        model.eval()
        print('Evaluating A model...')
        for _, (x, target) in enumerate(loader):
            #Move all data to appropriate device
            x = x.to(device=device, dtype=tensor_type)
            q = model.forward(x).squeeze().cpu()
            q_est.append(q)
            q_target.append(target.cpu())
            A_pred.append(model.output_A(x).cpu())
            
    A_pred = torch.cat(A_pred, dim=0)
    q_est = torch.cat(q_est, dim=0)
    q_target = torch.cat(q_target, dim=0)
    
    return (A_pred, q_est, q_target)

def wigner_log_likelihood(A, reduce=False):
    el, _ = np.linalg.eig(A)
    el.sort(axis=1)
    spacings = np.diff(el, axis=1)
    lls = np.log(spacings) - 0.25*np.pi*(spacings**2)
    if reduce:
        return np.sum(lls, axis=1).mean()
    else:
        return np.sum(lls, axis=1)

def ll_threshold(A, quantile=0.75):
    stats = wigner_log_likelihood(A)
    return np.quantile(stats, quantile)



def collect_vo_errors(saved_file):
    checkpoint = torch.load(saved_file)
    args = checkpoint['args']
    seqs_base_path = '/media/m2-drive/datasets/KITTI/single_files'
    if args.megalith:
        seqs_base_path = '/media/datasets/KITTI/single_files'
    seq_prefix = 'seq_'
    kitti_data_pickle_file = 'kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_True_minta_0.0.pickle'.format(args.seq)

    valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=None, run_type='test', seq_prefix=seq_prefix),
                                batch_size=args.batch_size_test, pin_memory=False,
                                shuffle=False, num_workers=args.num_workers, drop_last=False)
    T_21_vo = valid_loader.dataset.T_21_vo
    q_vo = torch.stack([rotmat_to_quat(torch.from_numpy(T[:3,:3])) for T in T_21_vo], dim=0)
    return q_vo

def collect_errors(saved_file, validation_transform=None):
    checkpoint = torch.load(saved_file)
    args = checkpoint['args']
    print(args)
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float

    seqs_base_path = '/media/m2-drive/datasets/KITTI/single_files'
    if args.megalith:
        seqs_base_path = '/media/datasets/KITTI/single_files'
    seq_prefix = 'seq_'
    kitti_data_pickle_file = 'kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_True_minta_0.0.pickle'.format(args.seq)

    train_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=None, run_type='train', seq_prefix=seq_prefix),
                                batch_size=args.batch_size_test, pin_memory=False,
                                shuffle=False, num_workers=args.num_workers, drop_last=False)

    valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=validation_transform, run_type='test', seq_prefix=seq_prefix),
                                batch_size=args.batch_size_test, pin_memory=False,
                                shuffle=False, num_workers=args.num_workers, drop_last=False)
    dim_in = 6

    

    if args.model == 'A_sym':
        model = QuatFlowNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        model.load_state_dict(checkpoint['model'], strict=False)
        A_predt, q_estt, q_targett = evaluate_A_model(train_loader, model, device, tensor_type)
        A_pred, q_est, q_target = evaluate_A_model(valid_loader, model, device, tensor_type)
        return ((A_predt, q_estt, q_targett), (A_pred, q_est, q_target))
    else:
        model = RotMat6DFlowNet(dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        model.load_state_dict(checkpoint['model'], strict=False)
        q_estt, q_targett = evaluate_rotmat_model(train_loader, model, device, tensor_type)
        q_est, q_target = evaluate_rotmat_model(valid_loader, model, device, tensor_type)
        return ((q_estt, q_targett), (q_est, q_target))

def create_kitti_data():

    prefix = 'saved_data/kitti/'
    file_list_6D = ['kitti_model_6D_seq_00_01-02-2020-14-21-01.pt', 'kitti_model_6D_seq_02_01-02-2020-15-13-10.pt','kitti_model_6D_seq_05_01-02-2020-16-09-34.pt']
    file_list_A_sym = ['kitti_model_A_sym_seq_00_01-01-2020-23-16-53.pt', 'kitti_model_A_sym_seq_02_01-02-2020-00-24-03.pt', 'kitti_model_A_sym_seq_05_01-01-2020-21-52-03.pt']
    
    print('Collecting normal data....')
    data_VO = []
    for file_A in file_list_A_sym:
        data_VO.append(collect_vo_errors(prefix + file_A))

    data_6D = []
    for file_6D in file_list_6D:
        data_6D.append(collect_errors(prefix + file_6D, validation_transform=None))

    data_A = []
    for file_A in file_list_A_sym:
        data_A.append(collect_errors(prefix + file_A, validation_transform=None))


    transform_erase_prob = 0.5
    transform = torchvision.transforms.RandomErasing(p=transform_erase_prob)
    print('Collecting transformed data....')
    data_6D_transformed = []
    for file_6D in file_list_6D:
        data_6D_transformed.append(collect_errors(prefix + file_6D, validation_transform=transform))

    data_A_transformed = []
    for file_A in file_list_A_sym:
        data_A_transformed.append(collect_errors(prefix + file_A, validation_transform=transform))

    print('Done')

    saved_data_file_name = 'kitti_comparison_data_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = 'saved_data/kitti/{}.pt'.format(saved_data_file_name)

    torch.save({
                'file_list_6D': file_list_6D,
                'file_list_A_sym': file_list_A_sym,
                'data_VO': data_VO,
                'data_6D': data_6D,
                'data_A': data_A,
                'data_6D_transformed': data_6D_transformed,
                'data_A_transformed': data_A_transformed,
                'transform_erase_prob': transform_erase_prob
    }, full_saved_path)

    print('Saved data to {}.'.format(full_saved_path))

    return

def _create_bar_plot(x_labels, bar_labels, heights, ylabel='mean error (deg)', xlabel='KITTI sequence', ylim=[0.05, 0.2]):
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots()
    fig.set_size_inches(4,2)

    x = np.arange(len(x_labels))
    N = len(bar_labels)
    colors = ['tab:red', 'tab:blue', 'tab:green']
    width = 0.4/N
    for i, (label, height) in enumerate(zip(bar_labels, heights)):
        ax.bar(x - 0.2 + width*i, height, width, label=label, color=colors[i], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)
    ax.legend(loc='upper right', fontsize = 8)
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    return fig

def _scatter(ax, x, y, title, color='tab:red', marker=".", size=4):
    ax.scatter(x, y, color=color, s=size, marker=marker, label=title)
    return

def _create_scatter_plot(thresh, lls, errors, labels, ylim=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(4,2)
    ax.axvline(thresh, c='k', ls='--', label='Threshold')
    colors = ['tab:blue', 'tab:red']
    for i, (ll, error, label) in enumerate(zip(lls, errors, labels)):
        _scatter(ax, ll, error, label, color=colors[i])
    ax.legend(loc='upper left')
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_ylabel('rotation error (deg)')
    ax.set_xlabel('Wigner log likelihood')
    ax.set_yscale('log')
    #ax.set_ylim(ylim)
    return fig

def create_plots():
    saved_data_file = 'saved_data/kitti/kitti_comparison_data_01-02-2020-19-28-23.pt'
    data = torch.load(saved_data_file)
    seqs = ['00', '02', '05']
    quantile = 0.75

    mean_err = []
    mean_err_filter = []
    mean_err_6D = []
    mean_err_vo = []
    
    mean_err_corrupted = []
    mean_err_corrupted_filter = []
    mean_err_corrupted_6D = []

    for s_i, seq in enumerate(seqs):
        (A_predt, q_estt, q_targett), (A_pred, q_est, q_target) = data['data_A'][s_i]
        mean_err.append(quat_angle_diff(q_est, q_target, reduce=True))
        thresh = ll_threshold(A_predt, quantile)
        mask = wigner_log_likelihood(A_pred) < thresh
        mean_err_filter.append(quat_angle_diff(q_est[mask], q_target[mask]))
        mean_err_vo.append(quat_angle_diff(data['data_VO'][s_i], q_target))
        
        #Create scatter plot
        fig = _create_scatter_plot(thresh, 
        [wigner_log_likelihood(A_predt), wigner_log_likelihood(A_pred)],
        [quat_angle_diff(q_estt, q_targett, reduce=False), quat_angle_diff(q_est, q_target, reduce=False)], labels=['Training', 'Validation'], ylim=[1e-4, 10])
        output_file = 'plots/kitti_scatter_seq_{}.pdf'.format(seq)
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)


        (q_estt, q_targett), (q_est, q_target) = data['data_6D'][s_i]
        mean_err_6D.append(quat_angle_diff(q_est, q_target, reduce=True))

        (A_predt, q_estt, q_targett), (A_pred, q_est, q_target) = data['data_A_transformed'][s_i]
        mean_err_corrupted.append(quat_angle_diff(q_est, q_target, reduce=True))
        thresh = ll_threshold(A_predt, quantile)
        mask = wigner_log_likelihood(A_pred) < thresh
        mean_err_corrupted_filter.append(quat_angle_diff(q_est[mask], q_target[mask]))

        #Create scatter plot
        fig = _create_scatter_plot(thresh, 
        [wigner_log_likelihood(A_predt), wigner_log_likelihood(A_pred)],
        [quat_angle_diff(q_estt, q_targett, reduce=False), quat_angle_diff(q_est, q_target, reduce=False)], labels=['Training', 'Validation'], ylim=[1e-4, 10])
        output_file = 'plots/kitti_scatter_seq_{}_corrupted.pdf'.format(seq)
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)

        (q_estt, q_targett), (q_est, q_target) = data['data_6D_transformed'][s_i]
        mean_err_corrupted_6D.append(quat_angle_diff(q_est, q_target, reduce=True))    

    bar_labels = ['VO','6D', 'A (Sym)', 'A (Sym) + Filter']
    fig = _create_bar_plot(seqs, bar_labels, [mean_err_vo, mean_err_6D, mean_err, mean_err_filter], ylim=[0,0.45])
    output_file = 'plots/kitti_normal.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

    bar_labels = ['6D', 'A (Sym)', 'A (Sym) + Filter']
    fig = _create_bar_plot(seqs, bar_labels, [mean_err_corrupted_6D, mean_err_corrupted, mean_err_corrupted_filter], xlabel='KITTI sequence (with corruption)', ylim=[0,0.45])
    output_file = 'plots/kitti_corrupted.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)



if __name__=='__main__':
    create_kitti_data()
    #create_plots()
