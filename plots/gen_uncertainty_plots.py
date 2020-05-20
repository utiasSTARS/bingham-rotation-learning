import numpy as np
import torchvision
import torch
import matplotlib
from matplotlib.colors import to_rgba

matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
plt.rc('axes', axisbelow=True)
import matplotlib.ticker as mtick
from matplotlib.colors import to_rgba
import sys
sys.path.insert(0,'..')
from quaternions import *
from networks import *
from helpers_train_test import *
from liegroups.numpy import SO3
from datetime import datetime
from convex_layers import *
from torch.utils.data import Dataset, DataLoader
from loaders import KITTIVODatasetPreTransformed
from metrics import *

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


def evaluate_6D_model(loader, model, device, tensor_type):
    q_est = []
    q_target = []
    six_vec = []

    with torch.no_grad():
        model.eval()
        print('Evaluating rotmat model...')
        for _, (x, target) in enumerate(loader):
            #Move all data to appropriate device
            x = x.to(device=device, dtype=tensor_type)
            out = model.net.forward(x).squeeze().cpu()
            q = rotmat_to_quat(sixdim_to_rotmat(out))

            six_vec.append(out)
            q_est.append(q)
            q_target.append(target.cpu())
            
    q_est = torch.cat(q_est, dim=0)
    q_target = torch.cat(q_target, dim=0)
    six_vec = torch.cat(six_vec, dim=0)
    
    return (six_vec, q_est, q_target)

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

def evaluate_autoenc(loader, model, device, tensor_type):
    l1_means = []
    loss_fn = torch.nn.L1Loss(reduction='none')
    with torch.no_grad():
        model.eval()
        print('Evaluating Auto Encoder model...')
        for _, (imgs, _) in enumerate(loader):
            #Move all data to appropriate device
            img = imgs[:,:3,:,:].to(device=device, dtype=tensor_type)
            img_out, code = model.forward(img)
            losses = loss_fn(img_out, img) #Bx1x224x224
            losses = losses.mean(dim=(1,2,3))
            l1_means.append(losses.cpu())
            
    l1_means = torch.cat(l1_means, dim=0)
    
    return l1_means


def collect_vo_errors(saved_file):
    checkpoint = torch.load(saved_file)
    args = checkpoint['args']
    seqs_base_path = '/media/m2-drive/datasets/KITTI/single_files'
    if args.megalith:
        seqs_base_path = '/media/datasets/KITTI/single_files'
    seq_prefix = 'seq_'
    kitti_data_pickle_file = '../experiments/kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_True_minta_0.0.pickle'.format(args.seq)

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
    kitti_data_pickle_file = '../experiments/kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_True_minta_0.0.pickle'.format(args.seq)

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
        six_vect, q_estt, q_targett = evaluate_6D_model(train_loader, model, device, tensor_type)
        six_vec, q_est, q_target = evaluate_6D_model(valid_loader, model, device, tensor_type)
        return ((six_vect, q_estt, q_targett), (six_vec, q_est, q_target))

def collect_autoencoder_stats(saved_file, validation_transform=None):
    checkpoint = torch.load(saved_file)
    args = checkpoint['args']
    print(args)
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.float
    transform = None
    seqs_base_path = '/media/m2-drive/datasets/KITTI/single_files'
    if args.megalith:
        seqs_base_path = '/media/datasets/KITTI/single_files'

    seq_prefix = 'seq_'

    kitti_data_pickle_file = '../experiments/kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_False_min_turn_0.0.pickle'.format(args.seq)


    train_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=False, seqs_base_path=seqs_base_path, transform_img=transform, run_type='train', seq_prefix=seq_prefix),
                            batch_size=args.batch_size_train, pin_memory=False,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)

    valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=False, seqs_base_path=seqs_base_path, transform_second_half_only=True, transform_img=validation_transform, run_type='test', seq_prefix=seq_prefix),
                            batch_size=args.batch_size_test, pin_memory=False,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)

    model = ComplexAutoEncoder(dim_in=3, dim_latent=args.dim_latent, dim_transition=args.dim_transition).to(device=device, dtype=tensor_type)
    model.load_state_dict(checkpoint['model'], strict=False)
    l1_meanst = evaluate_autoenc(train_loader, model, device, tensor_type)
    l1_means = evaluate_autoenc(valid_loader, model, device, tensor_type)
    return (l1_meanst, l1_means)

def create_kitti_autoencoder_data():

    print('Collecting KITTI autoencoder data....')
    base_dir = '../saved_data/kitti/'
    file_list = ['kitti_autoencoder_seq_00_01-27-2020-22-13-12.pt', 'kitti_autoencoder_seq_02_01-27-2020-22-48-15.pt', 'kitti_autoencoder_seq_05_01-27-2020-23-27-20.pt']
    seqs = ['00','02', '05']
    autoenc_l1_means = []
    autoenc_l1_means_corrupted = []
    
    for file in file_list:
        autoenc_l1_means.append(collect_autoencoder_stats(base_dir+file))

    transform = torchvision.transforms.RandomErasing(p=1, scale=(0.25, 0.5), ratio=(0.33, 3))

    for file in file_list:
        autoenc_l1_means_corrupted.append(collect_autoencoder_stats(base_dir+file, validation_transform=transform))


    saved_data_file_name = 'processed_autoenc_3seqs_withcorrupted_{}.pt'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = '../saved_data/kitti/{}'.format(saved_data_file_name)

    torch.save({
                'file_list': file_list,
                'seqs': seqs,
                'autoenc_l1_means': autoenc_l1_means,
                'autoenc_l1_means_corrupted': autoenc_l1_means_corrupted
    }, full_saved_path)

    print('Saved data to {}.'.format(full_saved_path))

    return full_saved_path


def create_table_stats_autoenc(Asym_data_file, autoenc_data_file):
    asym_data = torch.load(Asym_data_file)
    autoenc_data = torch.load(autoenc_data_file)
    seqs = ['00', '02', '05']
    quantile_ae = 1.0
    for i in range(3):
        l1_meanst = autoenc_data['autoenc_l1_means'][i][0]
        (q_est_6D, q_target_6D) = asym_data['data_6D'][i][1]

        mean_err_6D =  quat_angle_diff(q_est_6D, q_target_6D)
        thresh_ae = compute_threshold(l1_meanst.numpy(), uncertainty_metric_fn=l1_norm, quantile=quantile_ae)
        l1_means = autoenc_data['autoenc_l1_means'][i][1]
        mask_ae = compute_mask(l1_means.numpy(), l1_norm, thresh_ae)

        mean_err_6D_ae = quat_angle_diff(q_est_6D[mask_ae], q_target_6D[mask_ae])
        precision, recall = compute_prec_recall(l1_meanst.numpy(), l1_means.numpy(), quantile_ae, l1_norm)

        print('Seq {} 6D err: {:.2F} | 6D + AE (q: {}) err: {:.2F} | Kept: {:.1F}% | Precision: {:.2F}'.format(seqs[i], mean_err_6D, quantile_ae, mean_err_6D_ae, 100.*mask_ae.sum()/mask_ae.shape[0], 100.*precision))

    for i in range(3):
        l1_meanst = autoenc_data['autoenc_l1_means'][i][0]
        (q_est_6D, q_target_6D) = asym_data['data_6D_transformed'][i][1]

        mean_err_6D =  quat_angle_diff(q_est_6D, q_target_6D)
        thresh_ae = compute_threshold(l1_meanst.numpy(), uncertainty_metric_fn=l1_norm, quantile=quantile_ae)
        l1_means = autoenc_data['autoenc_l1_means_corrupted'][i][1]
        mask_ae = compute_mask(l1_means.numpy(), l1_norm, thresh_ae)
        mean_err_6D_ae = quat_angle_diff(q_est_6D[mask_ae], q_target_6D[mask_ae])


        precision, recall = compute_prec_recall(l1_meanst.numpy(), l1_means.numpy(), quantile_ae, l1_norm)

        print('Seq {} CORRUPT 6D err: {:.2F} | 6D + AE (q: {}) err: {:.2F} | Kept: {:.1F}% | Precision: {:.2F}'.format(seqs[i],mean_err_6D, quantile_ae, mean_err_6D_ae, 100.*mask_ae.sum()/mask_ae.shape[0], 100.*precision))

def create_bar_autoenc(Asym_data_file, autoenc_data_file):

    asym_data = torch.load(Asym_data_file)
    autoenc_data = torch.load(autoenc_data_file)
    quantile_ae = 1.0
    quantile_dt = 0.75

    
    mean_err_quat = []
    mean_err_A = []
    mean_err_A_dt = []
    mean_err_6D = []
    mean_err_6D_ae = []


    for i in range(3):

        (A_train, _, _)  = asym_data['data_A'][i][0]

        l1_meanst = autoenc_data['autoenc_l1_means'][i][0]
        
        (A_test, q_est, q_target) = asym_data['data_A'][i][1]
        (q_est_6D, q_target_6D) = asym_data['data_6D'][i][1]
        (q_est_quat, q_target_quat) = asym_data['data_quat'][i][1]

        mean_err_quat.append(quat_angle_diff(q_est_quat, q_target_quat))
        mean_err_A.append(quat_angle_diff(q_est, q_target))
        mean_err_6D.append(quat_angle_diff(q_est_6D, q_target_6D))

        thresh_ae = compute_threshold(l1_meanst.numpy(), uncertainty_metric_fn=l1_norm, quantile=quantile_ae)
        l1_means = autoenc_data['autoenc_l1_means'][i][1]
        mask_ae = compute_mask(l1_means.numpy(), l1_norm, thresh_ae)

        mean_err_6D_ae.append(quat_angle_diff(q_est_6D[mask_ae], q_target_6D[mask_ae]))
        
        thresh_dt = compute_threshold(A_train.numpy(), uncertainty_metric_fn=sum_bingham_dispersion_coeff, quantile=quantile_dt)
        mask_dt = compute_mask(A_test.numpy(), sum_bingham_dispersion_coeff, thresh_dt)
        
        mean_err_A_dt.append(quat_angle_diff(q_est[mask_dt], q_target[mask_dt]))


    dataset_names = ['00', '02', '05']
    bar_labels = ['\\texttt{6D}', '$\mathbf{A}$', '\\texttt{6D} + \\textit{AE} ' + '($q:$ {})'.format(quantile_ae), '$\mathbf{A}$ + \\textit{DT}' + ' ($q: {}$)'.format(quantile_dt)]
    fig = _create_bar_plot(dataset_names, bar_labels, [mean_err_6D, mean_err_A, mean_err_6D_ae, mean_err_A_dt], ylim=[0,0.5], xlabel='KITTI Dataset')
    output_file = 'kitti_autoenc_errors_bar.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    print('Outputted {}.'.format(output_file))

    #====================CORRUPTED==================
    mean_err_quat = []
    mean_err_A = []
    mean_err_A_dt = []
    mean_err_6D = []
    mean_err_6D_ae = []

    for i in range(3):

        (A_train, _, _)  = asym_data['data_A'][i][0]

        l1_meanst = autoenc_data['autoenc_l1_means'][i][0]
        
        (A_test, q_est, q_target) = asym_data['data_A_transformed'][i][1]
        (q_est_6D, q_target_6D) = asym_data['data_6D_transformed'][i][1]
        (q_est_quat, q_target_quat) = asym_data['data_quat_transformed'][i][1]

        mean_err_quat.append(quat_angle_diff(q_est_quat, q_target_quat))
        mean_err_A.append(quat_angle_diff(q_est, q_target))
        mean_err_6D.append(quat_angle_diff(q_est_6D, q_target_6D))

        thresh_ae = compute_threshold(l1_meanst.numpy(), uncertainty_metric_fn=l1_norm, quantile=quantile_ae)
        l1_means = autoenc_data['autoenc_l1_means_corrupted'][i][1]
        mask_ae = compute_mask(l1_means.numpy(), l1_norm, thresh_ae)

        mean_err_6D_ae.append(quat_angle_diff(q_est_6D[mask_ae], q_target_6D[mask_ae]))
        
        thresh_dt = compute_threshold(A_train.numpy(), uncertainty_metric_fn=sum_bingham_dispersion_coeff, quantile=quantile_dt)
        mask_dt = compute_mask(A_test.numpy(), sum_bingham_dispersion_coeff, thresh_dt)
        
        mean_err_A_dt.append(quat_angle_diff(q_est[mask_dt], q_target[mask_dt]))


    dataset_names = ['00', '02', '05']
    bar_labels = ['6D', 'A', '6D + AE (q: {})'.format(quantile_ae), 'A + DT (q: {})'.format(quantile_dt)]
    fig = _create_bar_plot(dataset_names, bar_labels, [mean_err_6D, mean_err_A, mean_err_6D_ae, mean_err_A_dt], ylim=[0,1], xlabel='KITTI Dataset', legend=False, ylabel=None)
    output_file = 'kitti_autoenc_errors_corrupted_bar.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    print('Outputted {}.'.format(output_file))

def create_kitti_data():

    prefix = '../saved_data/kitti/'
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

    saved_data_file_name = 'kitti_comparison_data_6Dvec_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = '../saved_data/kitti/{}.pt'.format(saved_data_file_name)

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
    # colors = ['tab:green','tab:red', 'tab:blue', 'tab:red', 'tab:blue']
    # alpha = [1.,1.,1.,0.5,0.5]
    colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue']
    alpha = [1.,1.,0.5,0.5]
    
    width = 0.5/N
    for i, (label, height) in enumerate(zip(bar_labels, heights)):
        ax.bar(x - 0.25 + width*i, height, width, label=label, color=to_rgba(colors[i], alpha=alpha[i]), linewidth=0.75, edgecolor=colors[i])
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

def _create_scatter_plot(thresh, thresh_label, lls, errors, labels, xlabel, ylim=None, legend=True, ylabel=True):
    fig, ax = plt.subplots()
    fig.set_size_inches(2,1.5)
    ax.axvline(thresh, c='k', ls='--', lw=0.75, label=thresh_label)
    colors = ['tab:orange','grey']
    markers = ['.', '+']
    for i, (ll, error, label) in enumerate(zip(lls, errors, labels)):
        _scatter(ax, ll, error, label, color=colors[i], size=1, marker=markers[i], rasterized=True)
    if legend:
        ax.legend(loc='upper left', markerscale=5.0)
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    if ylabel:
        ax.set_ylabel('error (deg)')
    ax.set_xlabel(xlabel)
    #ax.set_yscale('log')
    #ax.set_xscale('symlog')
    #ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    ax.set_ylim(ylim)
    return fig

def compute_prec_recall(A_train, A_test, quantile, uncertainty_metric_fn):
    thresh = compute_threshold(A_train, uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
    mask = compute_mask(A_test, uncertainty_metric_fn, thresh)

    true_mask = np.zeros(mask.shape)
    true_mask[:int(true_mask.shape[0]/2)] = 1.
    num_correct = int((true_mask*mask).sum())
    num_picked_out = mask.sum()
    precision = num_correct/(num_picked_out + 1e-8)
    recall = num_correct/true_mask.sum()
    return precision, recall

def create_precision_recall_plot(uncertainty_metric_fn, selected_quantile):
    saved_data_file = '../saved_data/kitti/kitti_comparison_data_01-04-2020-12-35-32.pt'
    data = torch.load(saved_data_file)
    seqs = ['00', '02', '05']
    colors = ['--k', '-.k', '-k']

    quantiles = np.arange(0.05, 0.98, 0.02)
    fig, ax = plt.subplots()
    fig.set_size_inches(4,1.5)

    for s_i, seq in enumerate(seqs):
        precision = np.empty(len(quantiles))
        recall = np.empty(len(quantiles))
        
        for q_i, quantile in enumerate(quantiles):
            (A_train, _, _), (A_test, _, _) = data['data_A_transformed'][s_i]
            precision[q_i], recall[q_i] = compute_prec_recall(A_train, A_test, quantile, uncertainty_metric_fn)

        _plot_curve(ax, recall, precision, 'Seq '+seq, colors[s_i])
        p, r = compute_prec_recall(A_train, A_test, selected_quantile, uncertainty_metric_fn)
        label = '{:.2F} quantile'.format(selected_quantile) if s_i == 0 else ''
        ax.scatter(r, p, label=label, marker='D', color='black', s=10, zorder=s_i)



    ax.legend(loc='lower left')
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_ylabel('precision')
    ax.set_xlabel('recall')
    #ax.set_ylim([0.7, 1])
    output_file = 'kitti_prec_recall.pdf'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


def create_table_stats(uncertainty_metric_fn=first_eig_gap):
    saved_data_file = '../saved_data/kitti/kitti_comparison_data_01-04-2020-12-35-32.pt'
    data = torch.load(saved_data_file)
    seqs = ['00', '02', '05']
    quantiles = [0.75, 0.5, 0.25]
    print('Uncertainty metric: {}'.format(uncertainty_metric_fn.__name__))
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
            thresh = compute_threshold(A_train.numpy(), uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
            mask = compute_mask(A_test.numpy(), uncertainty_metric_fn, thresh)

            mean_err_A_filter = quat_angle_diff(q_est[mask], q_target[mask])
            
            print('Quantile: {}. A (sym + Thresh): {:.2F} | Kept: {:.1F}%'.format(quantile, mean_err_A_filter, 100.*mask.sum()/mask.shape[0]))

        
        _, (q_est, q_target) = data['data_quat_transformed'][s_i]
        mean_err_quat = quat_angle_diff(q_est, q_target)

        _, (q_est, q_target) = data['data_6D_transformed'][s_i]
        mean_err_6D = quat_angle_diff(q_est, q_target)

        (A_train, _, _), (A_test, q_est, q_target) = data['data_A_transformed'][s_i]
        mean_err_A = quat_angle_diff(q_est, q_target)

        print('-- CORRUPTED --')
        print('Mean Error (deg): Quat: {:.2F} | 6D: {:.2F} | A (sym) {:.2F}'.format(mean_err_quat, mean_err_6D, mean_err_A))

        for q_i, quantile in enumerate(quantiles):
            thresh = compute_threshold(A_train.numpy(), uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
            mask = compute_mask(A_test.numpy(), uncertainty_metric_fn, thresh)
            mean_err_A_filter = quat_angle_diff(q_est[mask], q_target[mask])
            precision, recall = compute_prec_recall(A_train.numpy(), A_test.numpy(), quantile, uncertainty_metric_fn)

            print('Quantile: {}. A (sym + Thresh): {:.2F} | Kept: {:.1F}% | Precision: {:.2F}'.format(quantile, mean_err_A_filter, 100.*mask.sum()/mask.shape[0], 100.*precision))

def create_table_stats_6D():
    saved_data_file = '../saved_data/kitti/kitti_comparison_data_6Dvec_01-15-2020-16-19-36.pt'
    data = torch.load(saved_data_file)
    seqs = ['00', '02', '05']
    quantiles = [0.25, 0.5, 0.75]
    for s_i, seq in enumerate(seqs):

        (six_vec_train, _, _), (six_vec_test, q_est, q_target) = data['data_6D'][s_i]
        mean_err_6D = quat_angle_diff(q_est, q_target)

        print('Seq: {}. Total Pairs: {}.'.format(seq, q_est.shape[0]))
        
        print('Mean Error (deg): 6D: {:.2F}'.format(mean_err_6D))

        for q_i, quantile in enumerate(quantiles):
            thresh = compute_threshold(six_vec_train, uncertainty_metric_fn=l2_norm, quantile=quantile)
            mask = compute_mask(six_vec_test, l2_norm, thresh)
            mean_err_6D_filter = quat_angle_diff(q_est[mask], q_target[mask])
            
            print('Quantile: {}. 6D + Norm Filter: {:.2F} | Kept: {:.1F}%'.format(quantile, mean_err_6D_filter, 100.*mask.sum()/mask.shape[0]))


        (six_vec_train, _, _), (six_vec_test, q_est, q_target) = data['data_6D_transformed'][s_i]
        mean_err_6D = quat_angle_diff(q_est, q_target)


        print('-- CORRUPTED --')
        print('Mean Error (deg): 6D: {:.2F}'.format(mean_err_6D))

        for q_i, quantile in enumerate(quantiles):
            thresh = compute_threshold(six_vec_train, uncertainty_metric_fn=l2_norm, quantile=quantile)
            mask = compute_mask(six_vec_test, l2_norm, thresh)
            mean_err_6D_filter = quat_angle_diff(q_est[mask], q_target[mask])
            precision, recall = compute_prec_recall(six_vec_train, six_vec_test, quantile, l2_norm)

            print('Quantile: {}. 6D + Norm Filter: {:.2F} | Kept: {:.1F}% | Precision: {:.2F}%'.format(quantile, mean_err_6D_filter, 100.*mask.sum()/mask.shape[0], 100.*precision))


def create_box_plots(cache_data=True, uncertainty_metric_fn=first_eig_gap, logscale=False):
    if cache_data:
        prefix = '../saved_data/kitti/'
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
            kitti_data_pickle_file = '../experiments/kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_True_minta_0.0.pickle'.format(args.seq)
            train_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='train', seq_prefix=seq_prefix),
                                        batch_size=args.batch_size_test, pin_memory=False,
                                        shuffle=False, num_workers=args.num_workers, drop_last=False)
            valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test', seq_prefix=seq_prefix),
                                        batch_size=args.batch_size_test, pin_memory=False,
                                        shuffle=False, num_workers=args.num_workers, drop_last=False)
            kitti_data_pickle_file_new = '../experiments/kitti/kitti_singlefile_data_sequence_01_delta_1_reverse_True_min_turn_0.0.pickle'
            valid_loader2 = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file_new, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test', seq_prefix=seq_prefix),
                                        batch_size=args.batch_size_test, pin_memory=False,
                                        shuffle=False, num_workers=args.num_workers, drop_last=False)

            transform = torchvision.transforms.RandomErasing(p=1, scale=(0.25, 0.5), ratio=(0.33, 3))
            valid_loader3 = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test', seq_prefix=seq_prefix),
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
            A_pred3, _, _ = evaluate_A_model(valid_loader3, model, device, tensor_type)

            x_rand = torch.randn(1000, 6, 224, 224)
            A_randn = []
            with torch.no_grad():
                model.eval()
                print('Evaluating rand A model...')
                batch = 50
                iters = int(x_rand.shape[0] / batch)
                for k in range(iters):
                    #Move all data to appropriate device
                    start, end = k * batch, (k + 1) * batch
                    x = x_rand[start:end].to(device=device, dtype=tensor_type)
                    A_randn.append(model.output_A(x).cpu())
            A_randn = torch.cat(A_randn, dim=0)
            A_list.append((A_predt, A_pred, A_pred2, A_pred3, A_randn))
        
        saved_data_file_name = 'kitti_boxplot_data_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = '../saved_data/kitti/{}.pt'.format(saved_data_file_name)

        torch.save({
                    'A_list': A_list,
                    'file_list_A_sym': file_list_A_sym,
        }, full_saved_path)
        print('Saved data to {}.'.format(full_saved_path))
    else:
        full_saved_path = "../saved_data/kitti/kitti_boxplot_data_01-06-2020-13-26-15.pt"
    
    
    data = torch.load(full_saved_path)
    seqs = ['00', '02', '05']

    lw = 0.5
    boxprops = dict(linewidth=0.5, facecolor=to_rgba('tab:orange', alpha=0.4)) 
    whiskerprops = dict(linewidth=2*lw)
    flierprops = dict(marker='o', markersize=5,
                  markeredgewidth=lw)
    medianprops = dict(linewidth=lw, color='k')

    for i, As in enumerate(data['A_list']):
        fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
        fig.set_size_inches(5,1.5)
        bp = ax.boxplot([uncertainty_metric_fn(As[0]),uncertainty_metric_fn(As[1]), uncertainty_metric_fn(As[2]), uncertainty_metric_fn(As[3]), uncertainty_metric_fn(As[4])], 
        labels=['training \n (\\textit{res} / \\textit{city})', '\\texttt{seq ' + seqs[i] + '}\n (\\textit{res})', '\\texttt{seq 01} \n (\\textit{road})', '\\texttt{seq ' + seqs[i] + '}\n (\\textbf{corrupted})', 'random \n input'],
        patch_artist=True, flierprops=flierprops, notch=True, boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops
        )

        for p_i in range(len(bp['boxes'])):
            if p_i == 0:
                bp['boxes'][p_i].set(facecolor=to_rgba('tab:grey', alpha=0.4), edgecolor='k')
                bp['medians'][p_i].set(color='k')
                bp['fliers'][p_i].set(markeredgecolor='tab:grey')
            else:
                bp['boxes'][p_i].set(facecolor=to_rgba('tab:orange', alpha=0.4), edgecolor='tab:orange')
                bp['medians'][p_i].set(color='orange')
                bp['fliers'][p_i].set(markeredgecolor='tab:orange')

        for p_i in range(len(bp['whiskers'])):
            c_i = math.floor(p_i/2)
            if c_i==0:
                bp['whiskers'][p_i].set(color='tab:grey')
                bp['caps'][p_i].set(color='tab:grey')
            else:
                bp['whiskers'][p_i].set(color='tab:orange')
                bp['caps'][p_i].set(color='tab:orange')

            
            
        ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.set_ylabel(decode_metric_name(uncertainty_metric_fn)) 
        ax.set_yticklabels([])
        if logscale:
            ax.set_yscale('symlog')   
        output_file = 'kitti_box_seq_{}_metric_{}.pdf'.format(seqs[i], uncertainty_metric_fn.__name__)
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)

def create_bar_and_scatter_plots(output_scatter=True, uncertainty_metric_fn=first_eig_gap, quantile=0.25):
    saved_data_file = '../saved_data/kitti/kitti_comparison_data_01-04-2020-12-35-32.pt'
    data = torch.load(saved_data_file)
    seqs = ['00', '02', '05']

    mean_err = []
    mean_err_filter = []
    mean_err_6D = []
    mean_err_vo= []
    mean_err_quat = []
    
    mean_err_corrupted = []
    mean_err_corrupted_filter = []
    mean_err_corrupted_6D = []
    mean_err_corrupted_quat = []

    thresh_label = '\\textit{thresh} \n $q:' + '{:.2F}'.format(quantile) + '$'
    for s_i, seq in enumerate(seqs):
        (A_predt, q_estt, q_targett), (A_pred, q_est, q_target) = data['data_A'][s_i]
        mean_err.append(quat_angle_diff(q_est, q_target, reduce=True))
        thresh = compute_threshold(A_predt, uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
        mask = compute_mask(A_pred, uncertainty_metric_fn, thresh)

        mean_err_filter.append(quat_angle_diff(q_est[mask], q_target[mask]))
        mean_err_vo.append(quat_angle_diff(data['data_VO'][s_i], q_target))
        
        if output_scatter:
            #Create scatter plot
            fig = _create_scatter_plot(thresh, thresh_label,
            [uncertainty_metric_fn(A_pred), uncertainty_metric_fn(A_predt)],
            [quat_angle_diff(q_est, q_target, reduce=False), quat_angle_diff(q_estt, q_targett, reduce=False)], xlabel=decode_metric_name(uncertainty_metric_fn),labels=['test', 'train'], ylim=[1e-4, 5], legend=True)
            output_file = 'kitti_scatter_seq_{}_metric_{}.pdf'.format(seq, uncertainty_metric_fn.__name__)
            fig.savefig(output_file, bbox_inches='tight')
            plt.close(fig)


        (q_estt, q_targett), (q_est, q_target) = data['data_6D'][s_i]
        mean_err_6D.append(quat_angle_diff(q_est, q_target, reduce=True))

        (q_estt, q_targett), (q_est, q_target) = data['data_quat'][s_i]
        mean_err_quat.append(quat_angle_diff(q_est, q_target, reduce=True))

        (A_predt, q_estt, q_targett), (A_pred, q_est, q_target) = data['data_A_transformed'][s_i]
        mean_err_corrupted.append(quat_angle_diff(q_est, q_target, reduce=True))
        thresh = compute_threshold(A_predt, uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
        mask = compute_mask(A_pred, uncertainty_metric_fn, thresh)

        mean_err_corrupted_filter.append(quat_angle_diff(q_est[mask], q_target[mask]))
        
        true_mask = np.zeros(mask.shape)
        true_mask[:int(true_mask.shape[0]/2)] = 1.
        num_correct = int((true_mask*mask).sum())
        num_picked_out = mask.sum()
        print('{}/{} correct ({:.2F} precision, {:.2F} recall)'.format(num_correct,num_picked_out, num_correct/num_picked_out, num_correct/true_mask.sum()))
        
        if output_scatter:
            #Create scatter plot
            fig = _create_scatter_plot(thresh, thresh_label,
            [uncertainty_metric_fn(A_pred), uncertainty_metric_fn(A_predt)],
            [quat_angle_diff(q_est, q_target, reduce=False), quat_angle_diff(q_estt, q_targett, reduce=False)], xlabel=decode_metric_name(uncertainty_metric_fn), labels=['Test', 'Train'], ylim=[1e-4, 5], legend=False, ylabel=False)
            output_file = 'kitti_scatter_seq_{}_corrupted_metric_{}.pdf'.format(seq, uncertainty_metric_fn.__name__)
            fig.savefig(output_file, bbox_inches='tight')
            plt.close(fig)

        (q_estt, q_targett), (q_est, q_target) = data['data_6D_transformed'][s_i]
        mean_err_corrupted_6D.append(quat_angle_diff(q_est, q_target, reduce=True))    

        (q_estt, q_targett), (q_est, q_target) = data['data_quat_transformed'][s_i]
        mean_err_corrupted_quat.append(quat_angle_diff(q_est, q_target, reduce=True))    



    bar_labels = ['Quat', '6D', 'A', 'A \n DT (q: {:.2F})'.format(quantile)]
    fig = _create_bar_plot(seqs, bar_labels, [mean_err_quat, mean_err_6D, mean_err, mean_err_filter], ylim=[0,0.8])
    output_file = 'kitti_errors_metric_{}.pdf'.format(uncertainty_metric_fn.__name__)
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

    bar_labels = ['Quat', '6D', 'A', 'A \n DT (q: {:.2F})'.format(quantile)]
    fig = _create_bar_plot(seqs, bar_labels, [mean_err_corrupted_quat, mean_err_corrupted_6D, mean_err_corrupted, mean_err_corrupted_filter], ylim=[0,0.8], legend=False)
    output_file = 'kitti_corrupted_errors_metric_{}.pdf'.format(uncertainty_metric_fn.__name__)
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)



if __name__=='__main__':
    #create_kitti_data()
    uncertainty_metric_fn = sum_bingham_dispersion_coeff
    #create_bar_and_scatter_plots(output_scatter=True, uncertainty_metric_fn=uncertainty_metric_fn, quantile=0.75)
    #create_box_plots(cache_data=False, uncertainty_metric_fn=uncertainty_metric_fn, logscale=False)
    
    # create_precision_recall_plot(uncertainty_metric_fn, selected_quantile=0.75)
    # create_table_stats(uncertainty_metric_fn=uncertainty_metric_fn)

    #create_kitti_autoencoder_data()

    Asym_data_file = '../saved_data/kitti/kitti_comparison_data_01-04-2020-12-35-32.pt'
    autoenc_data_file = '../saved_data/kitti/processed_autoenc_3seqs_withcorrupted_01-27-2020-23-34-30.pt'
    create_bar_autoenc(Asym_data_file, autoenc_data_file)
    # create_table_stats_autoenc(Asym_data_file, autoenc_data_file)
    #create_table_stats_6D()
    # print("=================")
    # create_table_stats(sum_bingham_dispersion_coeff)