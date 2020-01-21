from quaternions import *
from networks import *
from helpers_train_test import *
from liegroups.numpy import SO3
import torch
from datetime import datetime
from convex_layers import *
from torch.utils.data import Dataset, DataLoader
from loaders import FLADataset
import torchvision.transforms as transforms
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

def wigner_log_likelihood_measure(A, reduce=False):
    el, _ = np.linalg.eig(A)
    el.sort(axis=1)
    spacings = np.diff(el, axis=1)
    lls = np.log(spacings) - 0.25*np.pi*(spacings**2)
    if reduce:
        return np.sum(lls, axis=1).mean()
    else:
        return np.sum(lls, axis=1)


def first_eig_gap(A):
    el = np.linalg.eigvalsh(A)
    spacings = np.diff(el, axis=1)
    return spacings[:, 0] 

def det_inertia_mat(A):
    #A_inertia = -A
    
    els = np.linalg.eigvalsh(A)

    els = els[:, 1:] + els[:, 0, None] 
    # min_el = els[:,0]
    # I = np.repeat(np.eye(4).reshape(1,4,4), A_inertia.shape[0], axis=0)
    # A_inertia = A_inertia + I*min_el[:,None,None]

    return els[:,0]*els[:,1]*els[:,2] #np.linalg.det(A_inertia)

def sum_bingham_dispersion_coeff(A):
    if len(A.shape) == 2:
        A = A.reshape(1,4,4)
    els = np.linalg.eigvalsh(A)
    min_el = els[:,0]
    I = np.repeat(np.eye(4).reshape(1,4,4), A.shape[0], axis=0)
    return np.trace(-A + I*min_el[:,None,None], axis1=1, axis2=2)

   
def l2_norm(vecs):
    return np.linalg.norm(vecs, axis=1)

def decode_metric_name(uncertainty_metric_fn):
    if uncertainty_metric_fn == first_eig_gap:
        return 'First Eigenvalue Gap'
    elif uncertainty_metric_fn == sum_bingham_dispersion_coeff:
        return 'Sum of Dispersion Coefficients'
    elif uncertainty_metric_fn == trace_inertia_mat:
        return 'Trace of Inertia Matrix (min eigvalue added)'
    else:
        raise ValueError('Unknown uncertainty metric')

def compute_threshold(A, uncertainty_metric_fn=first_eig_gap, quantile=0.75):
    #stats = wigner_log_likelihood(A)
    stats = uncertainty_metric_fn(A)
    return np.quantile(stats, quantile)

def compute_mask(A, uncertainty_metric_fn, thresh):
    if uncertainty_metric_fn == first_eig_gap:
        return uncertainty_metric_fn(A) > thresh
    elif uncertainty_metric_fn == sum_bingham_dispersion_coeff:
        return uncertainty_metric_fn(A) < thresh
    elif uncertainty_metric_fn == l2_norm:
        return uncertainty_metric_fn(A) > thresh
    else:
        raise ValueError('Unknown uncertainty metric')


def collect_errors(saved_file):
    checkpoint = torch.load(saved_file)
    args = checkpoint['args']
    print(args)
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float

    select_ids_train = checkpoint['select_ids_train']
    select_ids_test = checkpoint['select_ids_test']
    if args.megalith:
        dataset_dir = '/media/datasets/'
    else:
        dataset_dir = '/media/m2-drive/datasets/'

    image_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/flea3'
    pose_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/pose'
    transform = transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            transforms.ToTensor(),
    ])
    dim_in = 2

    train_loader = DataLoader(FLADataset(image_dir=image_dir, pose_dir=pose_dir, select_idx=select_ids_train, transform=transform),
                            batch_size=args.batch_size_train, pin_memory=False,
                            shuffle=True, num_workers=args.num_workers, drop_last=False)

    valid_loader = DataLoader(FLADataset(image_dir=image_dir, pose_dir=pose_dir, select_idx=select_ids_test, transform=transform, eval_mode=True),
                            batch_size=args.batch_size_test, pin_memory=False,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)

    

    if args.model == 'A_sym':
        model = QuatFlowNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        model.load_state_dict(checkpoint['model'], strict=False)
        A_predt, q_estt, q_targett = evaluate_A_model(train_loader, model, device, tensor_type)
        A_pred, q_est, q_target = evaluate_A_model(valid_loader, model, device, tensor_type)
        return ((A_predt, q_estt, q_targett), (A_pred, q_est, q_target))

def create_fla_data():

    print('Collecting data....')
    file_fla = 'saved_data/fla/fla_model_A_sym_01-20-2020-23-30-27.pt'
    data_fla = collect_errors(file_fla)

    saved_data_file_name = 'fla_comparison_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = 'saved_data/flat/{}.pt'.format(saved_data_file_name)

    torch.save({
                'file_fla': file_fla,
                'data_fla': data_fla
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

def _create_scatter_plot(thresh, lls, errors, labels, xlabel, ylim=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(4,1.5)
    ax.axvline(thresh, c='k', ls='--', label='Threshold')
    colors = ['tab:orange','grey']
    markers = ['.', '+']
    for i, (ll, error, label) in enumerate(zip(lls, errors, labels)):
        _scatter(ax, ll, error, label, color=colors[i], size=5, marker=markers[i], rasterized=True)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_ylabel('rotation error (deg)')
    ax.set_xlabel(xlabel)
    #ax.set_yscale('log')
    #ax.set_xscale('symlog')
    #ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    ax.set_ylim(ylim)
    return fig


def create_table_stats(uncertainty_metric_fn=first_eig_gap):
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
            thresh = compute_threshold(A_train.numpy(), uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
            mask = compute_mask(A_test.numpy(), uncertainty_metric_fn, thresh)

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
            thresh = compute_threshold(A_train.numpy(), uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
            mask = compute_mask(A_test.numpy(), uncertainty_metric_fn, thresh)
            mean_err_A_filter = quat_angle_diff(q_est[mask], q_target[mask])
            precision, recall = compute_prec_recall(A_train.numpy(), A_test.numpy(), quantile, uncertainty_metric_fn)

            print('Quantile: {}. A (sym + WLLT): {:.2F} | Kept: {:.1F}% | Precision: {:.2F}'.format(quantile, mean_err_A_filter, 100.*mask.sum()/mask.shape[0], 100.*precision))

def create_table_stats_6D():
    saved_data_file = 'saved_data/kitti/kitti_comparison_data_6Dvec_01-15-2020-16-19-36.pt'
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



def create_bar_and_scatter_plots(output_scatter=True, uncertainty_metric_fn=first_eig_gap, quantile=0.25):
    #saved_data_file = 'saved_data/kitti/kitti_comparison_data_01-03-2020-01-03-26.pt'
    #saved_data_file = 'saved_data/kitti/kitti_comparison_data_01-03-2020-19-19-50.pt'
    saved_data_file = 'saved_data/kitti/kitti_comparison_data_01-04-2020-12-35-32.pt'
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

    for s_i, seq in enumerate(seqs):
        (A_predt, q_estt, q_targett), (A_pred, q_est, q_target) = data['data_A'][s_i]
        mean_err.append(quat_angle_diff(q_est, q_target, reduce=True))
        thresh = compute_threshold(A_predt, uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
        mask = compute_mask(A_pred, uncertainty_metric_fn, thresh)

        mean_err_filter.append(quat_angle_diff(q_est[mask], q_target[mask]))
        mean_err_vo.append(quat_angle_diff(data['data_VO'][s_i], q_target))
        
        if output_scatter:
            #Create scatter plot
            fig = _create_scatter_plot(thresh, 
            [uncertainty_metric_fn(A_pred), uncertainty_metric_fn(A_predt)],
            [quat_angle_diff(q_est, q_target, reduce=False), quat_angle_diff(q_estt, q_targett, reduce=False)], xlabel=decode_metric_name(uncertainty_metric_fn),labels=['Validation', 'Training'], ylim=[1e-4, 5])
            output_file = 'plots/kitti_scatter_seq_{}_metric_{}.pdf'.format(seq, uncertainty_metric_fn.__name__)
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
            fig = _create_scatter_plot(thresh, 
            [uncertainty_metric_fn(A_pred), uncertainty_metric_fn(A_predt)],
            [quat_angle_diff(q_est, q_target, reduce=False), quat_angle_diff(q_estt, q_targett, reduce=False)], xlabel=decode_metric_name(uncertainty_metric_fn), labels=['Validation', 'Training'], ylim=[1e-4, 5])
            output_file = 'plots/kitti_scatter_seq_{}_corrupted_metric_{}.pdf'.format(seq, uncertainty_metric_fn.__name__)
            fig.savefig(output_file, bbox_inches='tight')
            plt.close(fig)

        (q_estt, q_targett), (q_est, q_target) = data['data_6D_transformed'][s_i]
        mean_err_corrupted_6D.append(quat_angle_diff(q_est, q_target, reduce=True))    

        (q_estt, q_targett), (q_est, q_target) = data['data_quat_transformed'][s_i]
        mean_err_corrupted_quat.append(quat_angle_diff(q_est, q_target, reduce=True))    



    bar_labels = ['Quat', '6D', 'A (Sym)', 'A (Sym) \n Thresh (q: {:.2F})'.format(quantile)]
    fig = _create_bar_plot(seqs, bar_labels, [mean_err_quat, mean_err_6D, mean_err, mean_err_filter], ylim=[0,0.8])
    output_file = 'plots/kitti_errors_metric_{}.pdf'.format(uncertainty_metric_fn.__name__)
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

    bar_labels = ['Quat', '6D', 'A (Sym)', 'A (Sym) \n Thresh (q: {:.2F})'.format(quantile)]
    fig = _create_bar_plot(seqs, bar_labels, [mean_err_corrupted_quat, mean_err_corrupted_6D, mean_err_corrupted, mean_err_corrupted_filter], ylim=[0,0.8], legend=False)
    output_file = 'plots/kitti_corrupted_errors_metric_{}.pdf'.format(uncertainty_metric_fn.__name__)
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)



if __name__=='__main__':
    create_fla_data()
    #uncertainty_metric_fn = det_inertia_mat
    #create_bar_and_scatter_plots(output_scatter=True, uncertainty_metric_fn=uncertainty_metric_fn, quantile=0.75)
    #create_box_plots(cache_data=False, uncertainty_metric_fn=uncertainty_metric_fn, logscale=True)
     
    #create_precision_recall_plot()
    #create_table_stats(uncertainty_metric_fn=uncertainty_metric_fn)
    #create_box_plots(cache_data=False)

    #create_table_stats_6D()
    # print("=================")
    # create_table_stats(sum_bingham_dispersion_coeff)