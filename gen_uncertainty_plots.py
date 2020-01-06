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
import matplotlib.ticker as mtick
from matplotlib.colors import to_rgba

def evaluate_model(loader, model, device, tensor_type, rotmat_output=False):
    q_est = []
    q_target = []
    
    with torch.no_grad():
        model.eval()
        print('Evaluating rotmat model...')
        for _, (x, target) in enumerate(loader):
            #Move all data to appropriate device
            x = x.to(device=device, dtype=tensor_type)
            if rotmat_output:
                q = rotmat_to_quat(model.forward(x).squeeze().cpu())
            else:
                q = model.forward(x).squeeze().cpu()
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

def wigner_log_likelihood_measure(A, reduce=False):
    el, _ = np.linalg.eig(A)
    el.sort(axis=1)
    spacings = np.diff(el, axis=1)
    lls = np.log(spacings) - 0.25*np.pi*(spacings**2)
    if reduce:
        return np.sum(lls, axis=1).mean()
    else:
        return np.sum(lls, axis=1)

def epistemic_measure(A):
    el, _ = np.linalg.eig(A)
    el.sort(axis=1)
    spacings = np.diff(el, axis=1)
    return spacings[:, 0] 

def compute_threshold(A, quantile=0.75):
    #stats = wigner_log_likelihood(A)
    stats = epistemic_measure(A.numpy())
    return np.quantile(stats, quantile)

def compute_mask(measure, thresh):
    return measure > thresh

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

    # if validation_transform is not None:
    #     output_sample_images = 4
    # else:
    #     output_sample_images = 0
    output_sample_images = 0

    valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, output_sample_images=output_sample_images, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_second_half_only=True, transform_img=validation_transform, run_type='test', seq_prefix=seq_prefix),
                                batch_size=args.batch_size_test, pin_memory=False,
                                shuffle=False, num_workers=args.num_workers, drop_last=False)
    dim_in = 6

    

    if args.model == 'A_sym':
        model = QuatFlowNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        model.load_state_dict(checkpoint['model'], strict=False)
        A_predt, q_estt, q_targett = evaluate_A_model(train_loader, model, device, tensor_type)
        A_pred, q_est, q_target = evaluate_A_model(valid_loader, model, device, tensor_type)
        return ((A_predt, q_estt, q_targett), (A_pred, q_est, q_target))
    elif args.model == 'quat':
        model = BasicCNN(dim_in=dim_in, dim_out=4, normalize_output=True, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        model.load_state_dict(checkpoint['model'], strict=False)
        q_estt, q_targett = evaluate_model(train_loader, model, device, tensor_type,rotmat_output=False)
        q_est, q_target = evaluate_model(valid_loader, model, device, tensor_type,rotmat_output=False)
        return ((q_estt, q_targett), (q_est, q_target))
    else:
        model = RotMat6DFlowNet(dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        model.load_state_dict(checkpoint['model'], strict=False)
        q_estt, q_targett = evaluate_model(train_loader, model, device, tensor_type,rotmat_output=True)
        q_est, q_target = evaluate_model(valid_loader, model, device, tensor_type, rotmat_output=True)
        return ((q_estt, q_targett), (q_est, q_target))

def create_kitti_data():

    prefix = 'saved_data/kitti/'
    file_list_6D = ['kitti_model_6D_seq_00_01-02-2020-14-21-01.pt', 'kitti_model_6D_seq_02_01-02-2020-15-13-10.pt','kitti_model_6D_seq_05_01-02-2020-16-09-34.pt']
    file_list_A_sym = ['kitti_model_A_sym_seq_00_01-01-2020-23-16-53.pt', 'kitti_model_A_sym_seq_02_01-02-2020-00-24-03.pt', 'kitti_model_A_sym_seq_05_01-01-2020-21-52-03.pt']
    file_list_quat = ['kitti_model_quat_seq_00_01-04-2020-02-56-31.pt','kitti_model_quat_seq_02_01-04-2020-03-47-36.pt', 'kitti_model_quat_seq_05_01-04-2020-04-43-22.pt']
    
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

    data_quat = []
    for file_quat in file_list_quat:
        data_quat.append(collect_errors(prefix + file_quat, validation_transform=None))

    transform_erase_prob = 1
    transform = torchvision.transforms.RandomErasing(p=1, scale=(0.25, 0.5), ratio=(0.33, 3))

    print('Collecting transformed data....')
    data_6D_transformed = []
    for file_6D in file_list_6D:
        data_6D_transformed.append(collect_errors(prefix + file_6D, validation_transform=transform))

    data_A_transformed = []
    for file_A in file_list_A_sym:
        data_A_transformed.append(collect_errors(prefix + file_A, validation_transform=transform))

    data_quat_transformed = []
    for file_quat in file_list_quat:
        data_quat_transformed.append(collect_errors(prefix + file_quat, validation_transform=transform))

    print('Done')

    saved_data_file_name = 'kitti_comparison_data_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = 'saved_data/kitti/{}.pt'.format(saved_data_file_name)

    torch.save({
                'file_list_6D': file_list_6D,
                'file_list_A_sym': file_list_A_sym,
                'file_list_quat': file_list_quat,
                'data_quat': data_quat,
                'data_VO': data_VO,
                'data_6D': data_6D,
                'data_A': data_A,
                'data_6D_transformed': data_6D_transformed,
                'data_A_transformed': data_A_transformed,
                'data_quat_transformed': data_quat_transformed,
                'transform_erase_prob': transform_erase_prob
    }, full_saved_path)

    print('Saved data to {}.'.format(full_saved_path))

    return



def _create_bar_plot(x_labels, bar_labels, heights, ylabel='mean error (deg)', xlabel='KITTI sequence', ylim=[0., 0.8], legend=True):
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots()
    fig.set_size_inches(2,2)
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)

    x = np.arange(len(x_labels))
    N = len(bar_labels)
    colors = ['tab:green', 'tab:red', 'tab:blue', 'black']
    width = 0.5/N
    for i, (label, height) in enumerate(zip(bar_labels, heights)):
        ax.bar(x - 0.25 + width*i, height, width, label=label, color=to_rgba(colors[i], alpha=0.5), edgecolor=colors[i], linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)
    if legend:
        ax.legend(loc='upper right', fontsize = 8)
    return fig

def _scatter(ax, x, y, title, color='tab:red', marker=".", size =4, rasterized=False):
    ax.scatter(x, y, color=color, s=size, marker=marker, label=title, rasterized=rasterized)
    return

def _plot_curve(ax, x, y, label, style):
    ax.plot(x, y,  style, linewidth=1., label=label)
    return

def _create_scatter_plot(thresh, lls, errors, labels, ylim=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(4,2)
    ax.axvline(thresh, c='k', ls='--', label='Threshold')
    colors = ['grey', 'tab:orange']
    markers = ['.', '+']
    for i, (ll, error, label) in enumerate(zip(lls, errors, labels)):
        _scatter(ax, ll, error, label, color=colors[i], size=5, marker=markers[i], rasterized=True)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_ylabel('rotation error (deg)')
    ax.set_xlabel('first eigenvalue gap')
    ax.set_yscale('log')
    #ax.set_xscale('symlog')
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    #ax.set_ylim(ylim)
    return fig

def compute_prec_recall(A_train, A_test, quantile):
    thresh = compute_threshold(A_train, quantile)
    mask = compute_mask(epistemic_measure(A_test), thresh)

    true_mask = np.zeros(mask.shape)
    true_mask[:int(true_mask.shape[0]/2)] = 1.
    num_correct = int((true_mask*mask).sum())
    num_picked_out = mask.sum()
    precision = num_correct/(num_picked_out + 1e-8)
    recall = num_correct/true_mask.sum()
    return precision, recall

def create_precision_recall_plot():
    saved_data_file = 'saved_data/kitti/kitti_comparison_data_01-04-2020-12-35-32.pt'
    data = torch.load(saved_data_file)
    seqs = ['00', '02', '05']
    colors = ['--k', '-.k', '-k']

    quantiles = np.arange(0.01, 0.98, 0.02)
    selected_quantile = 0.25
    fig, ax = plt.subplots()
    fig.set_size_inches(4,1.5)

    for s_i, seq in enumerate(seqs):
        precision = np.empty(len(quantiles))
        recall = np.empty(len(quantiles))
        
        for q_i, quantile in enumerate(quantiles):
            (A_train, _, _), (A_test, _, _) = data['data_A_transformed'][s_i]
            precision[q_i], recall[q_i] = compute_prec_recall(A_train, A_test, quantile)

        _plot_curve(ax, recall, precision, 'Seq '+seq, colors[s_i])
        p, r = compute_prec_recall(A_train, A_test, selected_quantile)
        label = '{:.2F} quantile'.format(selected_quantile) if s_i == 0 else ''
        ax.scatter(r, p, label=label, marker='D', color='black', s=10, zorder=s_i)



    ax.legend(loc='lower left')
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_ylabel('precision')
    ax.set_xlabel('recall')
    #ax.set_ylim([0.7, 1])
    output_file = 'plots/kitti_prec_recall.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


def create_table_stats():
    saved_data_file = 'saved_data/kitti/kitti_comparison_data_01-04-2020-12-35-32.pt'
    data = torch.load(saved_data_file)
    seqs = ['00', '02', '05']
    quantiles = [0.25, 0.5, 0.75]
    for s_i, seq in enumerate(seqs):
        _, (q_est, q_target) = data['data_quat'][s_i]
        mean_err_quat = quat_angle_diff(q_est, q_target)

        _, (q_est, q_target) = data['data_6D'][s_i]
        mean_err_6D = quat_angle_diff(q_est, q_target)

        (A_train, _, _), (A_test, q_est, q_target) = data['data_A'][s_i]
        mean_err_A = quat_angle_diff(q_est, q_target)

        print('Seq: {}. Total Pairs: {}.'.format(seq, q_est.shape[0]))
        
        print('Mean Error (deg): Quat: {:.2F} | 6D: {:.2F} | A (sym) {:.2F}'.format(mean_err_quat, mean_err_6D, mean_err_A))

        for q_i, quantile in enumerate(quantiles):
            thresh = compute_threshold(A_train, quantile)
            mask = compute_mask(epistemic_measure(A_test), thresh)
            mean_err_A_filter = quat_angle_diff(q_est[mask], q_target[mask])
            
            print('Quantile: {}. A (sym + WLLT): {:.2F} | Kept: {:.1F}%'.format(quantile, mean_err_A_filter, 100.*mask.sum()/mask.shape[0]))

        
        _, (q_est, q_target) = data['data_quat_transformed'][s_i]
        mean_err_quat = quat_angle_diff(q_est, q_target)

        _, (q_est, q_target) = data['data_6D_transformed'][s_i]
        mean_err_6D = quat_angle_diff(q_est, q_target)

        (A_train, _, _), (A_test, q_est, q_target) = data['data_A_transformed'][s_i]
        mean_err_A = quat_angle_diff(q_est, q_target)

        print('-- CORRUPTED --')
        print('Mean Error (deg): Quat: {:.2F} | 6D: {:.2F} | A (sym) {:.2F}'.format(mean_err_quat, mean_err_6D, mean_err_A))

        for q_i, quantile in enumerate(quantiles):
            thresh = compute_threshold(A_train, quantile)
            mask = compute_mask(epistemic_measure(A_test), thresh)
            mean_err_A_filter = quat_angle_diff(q_est[mask], q_target[mask])
            precision, recall = compute_prec_recall(A_train, A_test, quantile)

            print('Quantile: {}. A (sym + WLLT): {:.2F} | Kept: {:.1F}% | Precision: {:.2F}'.format(quantile, mean_err_A_filter, 100.*mask.sum()/mask.shape[0], 100.*precision))

def create_box_plots(cache_data=True):
    if cache_data:
        prefix = 'saved_data/kitti/'
        file_list_A_sym = ['kitti_model_A_sym_seq_00_01-01-2020-23-16-53.pt', 'kitti_model_A_sym_seq_02_01-02-2020-00-24-03.pt', 'kitti_model_A_sym_seq_05_01-01-2020-21-52-03.pt']
        A_list = []
        for path in file_list_A_sym:
            checkpoint = torch.load(prefix+path)
            args = checkpoint['args']
            print(args)
            device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
            tensor_type = torch.double if args.double else torch.float

            transform = None
            seqs_base_path = '/media/m2-drive/datasets/KITTI/single_files'
            if args.megalith:
                seqs_base_path = '/media/datasets/KITTI/single_files'
            seq_prefix = 'seq_'
            kitti_data_pickle_file = 'kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_True_minta_0.0.pickle'.format(args.seq)
            train_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='train', seq_prefix=seq_prefix),
                                        batch_size=args.batch_size_test, pin_memory=False,
                                        shuffle=False, num_workers=args.num_workers, drop_last=False)
            valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test', seq_prefix=seq_prefix),
                                        batch_size=args.batch_size_test, pin_memory=False,
                                        shuffle=False, num_workers=args.num_workers, drop_last=False)
            kitti_data_pickle_file_new = 'kitti/kitti_singlefile_data_sequence_01_delta_1_reverse_True_min_turn_0.0.pickle'
            valid_loader2 = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file_new, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test', seq_prefix=seq_prefix),
                                        batch_size=args.batch_size_test, pin_memory=False,
                                        shuffle=False, num_workers=args.num_workers, drop_last=False)
            dim_in = 6
            model = QuatFlowNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
            valid_loader.dataset.rotmat_targets = False
            valid_loader2.dataset.rotmat_targets = False
            model.load_state_dict(checkpoint['model'], strict=False)

            #Train and test with new representation
            A_predt, _, _ = evaluate_A_model(train_loader, model, device, tensor_type)
            A_pred, _, _ = evaluate_A_model(valid_loader, model, device, tensor_type)
            A_pred2, _, _ = evaluate_A_model(valid_loader2, model, device, tensor_type)

            x = torch.randn(1000, 6, 224, 224)
            x = x.to(device=device, dtype=tensor_type)
            model.eval()
            with torch.no_grad():
                A_randn = model.output_A(x).cpu()
            A_list.append((A_predt, A_pred, A_pred2, A_randn))
        
        saved_data_file_name = 'kitti_boxplot_data_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/kitti/{}.pt'.format(saved_data_file_name)

        torch.save({
                    'A_list': A_list,
                    'file_list_A_sym': file_list_A_sym,
        }, full_saved_path)
        print('Saved data to {}.'.format(full_saved_path))
    else:
        full_saved_path = ""
    
    
    data = torch.load(full_saved_path)
    seqs = ['00', '02', '05']

    for i, As in enumerate(data['A_list']):
        fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
        ax.boxplot([epistemic_measure(As[0]),epistemic_measure(As[1]), epistemic_measure(As[2]), epistemic_measure(As[3])], 
        labels=['Training (Residential / City)', 'Seq ' + seqs[i] + ' (Residential)', 'Seq 01 (Road)', 'Random Input'])
        ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_ylabel('first eigenvalue gap')    
        output_file = 'plots/kitti_box_seq_{}.pdf'.format(seqs[i])
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)

def create_bar_and_scatter_plots(output_scatter=True):
    #saved_data_file = 'saved_data/kitti/kitti_comparison_data_01-03-2020-01-03-26.pt'
    #saved_data_file = 'saved_data/kitti/kitti_comparison_data_01-03-2020-19-19-50.pt'
    saved_data_file = 'saved_data/kitti/kitti_comparison_data_01-04-2020-12-35-32.pt'
    data = torch.load(saved_data_file)
    seqs = ['00', '02', '05']
    quantile = 0.25

    mean_err = []
    mean_err_filter = []
    mean_err_6D = []
    mean_err_vo= []
    mean_err_quat = []
    
    mean_err_corrupted = []
    mean_err_corrupted_filter = []
    mean_err_corrupted_6D = []
    mean_err_corrupted_quat = []

    for s_i, seq in enumerate(seqs):
        (A_predt, q_estt, q_targett), (A_pred, q_est, q_target) = data['data_A'][s_i]
        mean_err.append(quat_angle_diff(q_est, q_target, reduce=True))
        thresh = compute_threshold(A_predt, quantile)
        mask = compute_mask(epistemic_measure(A_pred), thresh)
        mean_err_filter.append(quat_angle_diff(q_est[mask], q_target[mask]))
        mean_err_vo.append(quat_angle_diff(data['data_VO'][s_i], q_target))
        
        if output_scatter:
            #Create scatter plot
            fig = _create_scatter_plot(thresh, 
            [epistemic_measure(A_predt), epistemic_measure(A_pred)],
            [quat_angle_diff(q_estt, q_targett, reduce=False), quat_angle_diff(q_est, q_target, reduce=False)], labels=['Training', 'Validation'], ylim=[1e-4, 10])
            output_file = 'plots/kitti_scatter_seq_{}.pdf'.format(seq)
            fig.savefig(output_file, bbox_inches='tight')
            plt.close(fig)


        (q_estt, q_targett), (q_est, q_target) = data['data_6D'][s_i]
        mean_err_6D.append(quat_angle_diff(q_est, q_target, reduce=True))

        (q_estt, q_targett), (q_est, q_target) = data['data_quat'][s_i]
        mean_err_quat.append(quat_angle_diff(q_est, q_target, reduce=True))

        (A_predt, q_estt, q_targett), (A_pred, q_est, q_target) = data['data_A_transformed'][s_i]
        mean_err_corrupted.append(quat_angle_diff(q_est, q_target, reduce=True))
        thresh = compute_threshold(A_predt, quantile)
        mask = compute_mask(epistemic_measure(A_pred), thresh)

        mean_err_corrupted_filter.append(quat_angle_diff(q_est[mask], q_target[mask]))
        
        true_mask = np.zeros(mask.shape)
        true_mask[:int(true_mask.shape[0]/2)] = 1.
        num_correct = int((true_mask*mask).sum())
        num_picked_out = mask.sum()
        print('{}/{} correct ({:.2F} precision, {:.2F} recall)'.format(num_correct,num_picked_out, num_correct/num_picked_out, num_correct/true_mask.sum()))
        
        if output_scatter:
            #Create scatter plot
            fig = _create_scatter_plot(thresh, 
            [epistemic_measure(A_predt), epistemic_measure(A_pred)],
            [quat_angle_diff(q_estt, q_targett, reduce=False), quat_angle_diff(q_est, q_target, reduce=False)], labels=['Training', 'Validation'], ylim=[1e-4, 10])
            output_file = 'plots/kitti_scatter_seq_{}_corrupted.pdf'.format(seq)
            fig.savefig(output_file, bbox_inches='tight')
            plt.close(fig)

        (q_estt, q_targett), (q_est, q_target) = data['data_6D_transformed'][s_i]
        mean_err_corrupted_6D.append(quat_angle_diff(q_est, q_target, reduce=True))    

        (q_estt, q_targett), (q_est, q_target) = data['data_quat_transformed'][s_i]
        mean_err_corrupted_quat.append(quat_angle_diff(q_est, q_target, reduce=True))    



    bar_labels = ['Quat', '6D', 'A (Sym)', 'A (Sym) \n EGT (q: {:.2F})'.format(quantile)]
    fig = _create_bar_plot(seqs, bar_labels, [mean_err_quat, mean_err_6D, mean_err, mean_err_filter], ylim=[0,0.8])
    output_file = 'plots/kitti_normal.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

    bar_labels = ['Quat', '6D', 'A (Sym)', 'A (Sym) \n EGT (q: {:.2F})'.format(quantile)]
    fig = _create_bar_plot(seqs, bar_labels, [mean_err_corrupted_quat, mean_err_corrupted_6D, mean_err_corrupted, mean_err_corrupted_filter], ylim=[0,0.8], legend=False)
    output_file = 'plots/kitti_corrupted.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)



if __name__=='__main__':
    #create_kitti_data()
    #create_bar_and_scatter_plots(output_scatter=False)
    #create_precision_recall_plot()
    #create_table_stats()
    create_box_plots(cache_data=True)