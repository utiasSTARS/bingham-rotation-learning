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
from cv2 import VideoWriter, VideoWriter_fourcc
import sys
sys.path.insert(0,'..')
from quaternions import *
from networks import *
from helpers_train_test import *
from liegroups.numpy import SO3
import torch
from datetime import datetime
from convex_layers import *
from torch.utils.data import Dataset, DataLoader
from loaders import FLADataset

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

def evaluate_autoenc(loader, model, device, tensor_type):
    l1_means = []
    loss_fn = torch.nn.L1Loss(reduction='none')
    with torch.no_grad():
        model.eval()
        print('Evaluating Auto Encoder model...')
        for _, (imgs, _) in enumerate(loader):
            #Move all data to appropriate device
            img = imgs[:,[0],:,:].to(device=device, dtype=tensor_type)
            img_out, code = model.forward(img)
            losses = loss_fn(img_out, img).mean(dim=(1,2))
            l1_means.append(losses.cpu())
            
    l1_means = torch.cat(l1_means, dim=0)
    
    return l1_means


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

    els = els[:, 1:] - els[:, 0, None] 
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
        return 'tr($\mathbf{\Lambda}$)'
    elif uncertainty_metric_fn == det_inertia_mat:
        return 'Det of Inertia Matrix (min eigvalue added)'
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
    if args.megalith:
        dataset_dir = '/media/datasets/'
    else:
        dataset_dir = '/media/m2-drive/datasets/'

    image_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/flea3'
    pose_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/pose'
    
    normalize = transforms.Normalize(mean=[0.45],
                                    std=[0.25])

    transform = transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    dim_in = 2

    train_dataset = '../experiments/FLA/{}_train.csv'.format(args.scene)

    train_loader = DataLoader(FLADataset(train_dataset, image_dir=image_dir, pose_dir=pose_dir, transform=transform),
                            batch_size=args.batch_size_train, pin_memory=False,
                            shuffle=True, num_workers=args.num_workers, drop_last=False)


    test_outdoor = FLADataset('../experiments/FLA/outdoor_test.csv', image_dir=image_dir, pose_dir=pose_dir, transform=transform)
    test_indoor = FLADataset('../experiments/FLA/indoor_test.csv', image_dir=image_dir, pose_dir=pose_dir, transform=transform)
    test_transition = FLADataset('../experiments/FLA/transition.csv', image_dir=image_dir, pose_dir=pose_dir, transform=transform)
    #valid_dataset = torch.utils.data.ConcatDataset([valid_dataset1, valid_dataset2, valid_dataset3])
    #test_dataset = FLADataset('FLA/{}_test.csv'.format(args.scene), image_dir=image_dir, pose_dir=pose_dir, transform=transform)
    
    valid_dataset1 = torch.utils.data.ConcatDataset([test_outdoor])
    valid_dataset2 = torch.utils.data.ConcatDataset([test_outdoor, test_indoor])
    valid_dataset3 = torch.utils.data.ConcatDataset([test_outdoor, test_indoor, test_transition])

    valid_loader1 = DataLoader(valid_dataset1,
                        batch_size=args.batch_size_test, pin_memory=True,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    valid_loader2 = DataLoader(valid_dataset2,
                        batch_size=args.batch_size_test, pin_memory=True,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    valid_loader3 = DataLoader(valid_dataset3,
                        batch_size=args.batch_size_test, pin_memory=True,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)


        

    if args.model == 'A_sym':
        model = QuatFlowNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        model.load_state_dict(checkpoint['model'], strict=False)
        A_predt, q_estt, q_targett = evaluate_A_model(train_loader, model, device, tensor_type)
        A_pred1, q_est1, q_target1 = evaluate_A_model(valid_loader1, model, device, tensor_type)
        A_pred2, q_est2, q_target2 = evaluate_A_model(valid_loader2, model, device, tensor_type)
        A_pred3, q_est3, q_target3 = evaluate_A_model(valid_loader3, model, device, tensor_type)
        return ((A_predt, q_estt, q_targett), (A_pred1, q_est1, q_target1), (A_pred2, q_est2, q_target2), (A_pred3, q_est3, q_target3))

def create_fla_data():

    print('Collecting data....')
    base_dir = '../saved_data/fla/'
    file_fla = 'fla_model_outdoor_A_sym_01-21-2020-15-45-02.pt'
    #file_fla = 'fla_model_indoor_A_sym_01-21-2020-15-54-30.pt'

    data_fla = collect_errors(base_dir+file_fla)

    saved_data_file_name = 'processed_3tests_{}'.format(file_fla)
    full_saved_path = '../saved_data/fla/{}'.format(saved_data_file_name)

    torch.save({
                'file_fla': file_fla,
                'data_fla': data_fla
    }, full_saved_path)

    print('Saved data to {}.'.format(full_saved_path))

    return full_saved_path


def collect_autoencoder_stats(saved_file):
    checkpoint = torch.load(saved_file)
    args = checkpoint['args']
    print(args)
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.float
    if args.megalith:
        dataset_dir = '/media/datasets/'
    else:
        dataset_dir = '/media/m2-drive/datasets/'

    image_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/flea3'
    pose_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/pose'
    
    normalize = transforms.Normalize(mean=[0.45],
                                    std=[0.25])

    transform = transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    dim_in = 2

    train_dataset = '../experiments/FLA/{}_train_reverse_False.csv'.format(args.scene)

    train_loader = DataLoader(FLADataset(train_dataset, image_dir=image_dir, pose_dir=pose_dir, transform=transform),
                            batch_size=args.batch_size_train, pin_memory=False,
                            shuffle=True, num_workers=args.num_workers, drop_last=False)


    test_outdoor = FLADataset('../experiments/FLA/outdoor_test.csv', image_dir=image_dir, pose_dir=pose_dir, transform=transform)
    test_indoor = FLADataset('../experiments/FLA/indoor_test.csv', image_dir=image_dir, pose_dir=pose_dir, transform=transform)
    test_transition = FLADataset('../experiments/FLA/transition.csv', image_dir=image_dir, pose_dir=pose_dir, transform=transform)
    
    valid_dataset1 = torch.utils.data.ConcatDataset([test_outdoor])
    valid_dataset2 = torch.utils.data.ConcatDataset([test_outdoor, test_indoor])
    valid_dataset3 = torch.utils.data.ConcatDataset([test_outdoor, test_indoor, test_transition])

    valid_loader1 = DataLoader(valid_dataset1,
                        batch_size=args.batch_size_test, pin_memory=True,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    valid_loader2 = DataLoader(valid_dataset2,
                        batch_size=args.batch_size_test, pin_memory=True,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    valid_loader3 = DataLoader(valid_dataset3,
                        batch_size=args.batch_size_test, pin_memory=True,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
        

    model = ComplexAutoEncoder(dim_in=1, dim_latent=args.dim_latent, dim_transition=args.dim_transition).to(device=device, dtype=tensor_type)
    model.load_state_dict(checkpoint['model'], strict=False)
    l1_meanst = evaluate_autoenc(train_loader, model, device, tensor_type)
    l1_means1 = evaluate_autoenc(valid_loader1, model, device, tensor_type)
    l1_means2 = evaluate_autoenc(valid_loader2, model, device, tensor_type)
    l1_means3 = evaluate_autoenc(valid_loader3, model, device, tensor_type)
    return (l1_meanst, l1_means1, l1_means2, l1_means3)


def create_fla_autoencoder_data():

    print('Collecting autoencoder data....')
    base_dir = '../saved_data/fla/'
    file_fla = 'fla_autoencoder_model_outdoor_01-27-2020-16-36-29.pt'

    autoenc_l1_means = collect_autoencoder_stats(base_dir+file_fla)

    saved_data_file_name = 'processed_3tests_{}'.format(file_fla)
    full_saved_path = '../saved_data/fla/{}'.format(saved_data_file_name)

    torch.save({
                'file_fla': file_fla,
                'autoenc_l1_means': autoenc_l1_means
    }, full_saved_path)

    print('Saved data to {}.'.format(full_saved_path))

    return full_saved_path


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
    ax.legend(loc='upper left')
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_ylabel('rotation error (deg)')
    ax.set_xlabel(xlabel)
    #ax.set_yscale('log')
    #ax.set_xscale('symlog')
    #ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    ax.set_ylim(ylim)
    return fig


def create_table_stats(uncertainty_metric_fn=first_eig_gap, data_file=None):
    #data_file = 'saved_data/fla/fla_comparison_01-21-2020-00-33-12.pt'
    data = torch.load(data_file)
    quantiles = [0.25, 0.5, 0.75]

    (A_train, _, _)  = data['data_fla'][0]
   

    for i in range(3):
        (A_test, q_est, q_target) = data['data_fla'][i+1]
        mean_err_A = quat_angle_diff(q_est, q_target)

        print('Total Pairs: {}.'.format(q_est.shape[0]))
        print('Mean Error (deg): A (sym) {:.2F}'.format(mean_err_A))

        for q_i, quantile in enumerate(quantiles):
            thresh = compute_threshold(A_train.numpy(), uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
            mask = compute_mask(A_test.numpy(), uncertainty_metric_fn, thresh)

            mean_err_A_filter = quat_angle_diff(q_est[mask], q_target[mask])
            
            print('Quantile: {}. A (sym + thresh): {:.2F} | Kept: {:.1F}%'.format(quantile, mean_err_A_filter, 100.*mask.sum()/mask.shape[0]))

        


def create_bar_and_scatter_plots(uncertainty_metric_fn=first_eig_gap, quantile=0.25, data_file=None):
    data = torch.load(data_file)
    
    (A_predt, q_estt, q_targett), (A_pred, q_est, q_target) = data['data_fla']

    thresh = compute_threshold(A_predt.numpy(), uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
    mask = compute_mask(A_pred.numpy(), uncertainty_metric_fn, thresh)

    fig = _create_scatter_plot(thresh, 
    [uncertainty_metric_fn(A_pred.numpy()), uncertainty_metric_fn(A_predt.numpy())],
    [quat_angle_diff(q_est, q_target, reduce=False), quat_angle_diff(q_estt, q_targett, reduce=False)], xlabel=decode_metric_name(uncertainty_metric_fn),labels=['Validation', 'Training'], ylim=[1e-4, 5])
    
    desc = data_file.split('/')[-1].split('.pt')[0]
    output_file = 'fla_scatter_metric_{}_{}.pdf'.format(uncertainty_metric_fn.__name__, desc)
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


def create_video(full_data_file=None):
    
    if full_data_file is None:
        data_file = '../saved_data/fla/fla_model_outdoor_A_sym_01-21-2020-15-45-02.pt'
        checkpoint = torch.load(data_file)
        args = checkpoint['args']
        print(args)
        device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
        tensor_type = torch.double if args.double else torch.float
        if args.megalith:
            dataset_dir = '/media/datasets/'
        else:
            dataset_dir = '/media/m2-drive/datasets/'

        image_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/flea3'
        pose_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/pose'
        
        normalize = transforms.Normalize(mean=[0.45],
                                        std=[0.25])

        transform = transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ])
        dim_in = 2
        train_dataset = '../experiments/FLA/{}_train.csv'.format(args.scene)
        train_loader = DataLoader(FLADataset(train_dataset, image_dir=image_dir, pose_dir=pose_dir, transform=transform),
                                batch_size=args.batch_size_train, pin_memory=False,
                                shuffle=True, num_workers=args.num_workers, drop_last=False)


        valid_dataset = FLADataset('../experiments/FLA/all_moving_unshuffled.csv', image_dir=image_dir, pose_dir=pose_dir, transform=transform)
        #valid_dataset = torch.utils.data.ConcatDataset([valid_dataset1, valid_dataset2, valid_dataset3])
        #test_dataset = FLADataset('FLA/{}_test.csv'.format(args.scene), image_dir=image_dir, pose_dir=pose_dir, transform=transform)
        valid_loader = DataLoader(valid_dataset,
                            batch_size=args.batch_size_test, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)
        model = QuatFlowNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        model.load_state_dict(checkpoint['model'], strict=False)
        A_predt, q_estt, q_targett = evaluate_A_model(train_loader, model, device, tensor_type)
        A_pred, q_est, q_target = evaluate_A_model(valid_loader, model, device, tensor_type)
        data = ((A_predt, q_estt, q_targett), (A_pred, q_est, q_target))

        desc = data_file.split('/')[-1].split('.pt')[0]
        saved_data_file_name = 'processed_video_{}.pt'.format(desc)
        full_data_file = '../saved_data/fla/{}'.format(saved_data_file_name)
        torch.save({
                    'file_fla': data_file,
                    'data_fla': data
        }, full_data_file)

        print('Saved data to {}.'.format(full_data_file))

    else:
        data = torch.load(full_data_file) 
        quantile = 0.75
        uncertainty_metric_fn = sum_bingham_dispersion_coeff
        (A_train, _, _), (A_test, q_est, q_target) = data['data_fla']
        thresh = compute_threshold(A_train.numpy(), uncertainty_metric_fn=uncertainty_metric_fn, quantile=quantile)
        mask = compute_mask(A_test.numpy(), uncertainty_metric_fn, thresh)

        transform = transforms.ToTensor()
        dataset_dir = '/Users/valentinp/Dropbox/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/'
        image_dir = dataset_dir+'flea3'
        pose_dir = dataset_dir+'pose'
        
        all_dataset = FLADataset('../FLA/all_moving_unshuffled.csv', image_dir=image_dir, pose_dir=pose_dir, transform=transform)
        
        fourcc = VideoWriter_fourcc(*'MP4V')
        FPS = 60
        width = 640
        height = 512

        video_array = np.empty((len(all_dataset), height, width, 3))

        for i in range(len(all_dataset)):
            imgs, _ = all_dataset[i]
            img = imgs[0].numpy().reshape(height, width, 1)*255
            img = img.repeat(3, axis=2).astype(np.uint8)
            
            if mask[i] == 0:
                img[:100,:100, 0] = 255
                img[:100,:100, 1] = 0
                img[:100,:100, 2] = 0
                
            else:
                img[:100,:100, 0] = 0
                img[:100,:100, 1] = 255
                img[:100,:100, 2] = 0
                
            video_array[i] = img

            if i%1000==0:
                print(i)

        torchvision.io.video.write_video('fla.mp4', video_array, FPS, video_codec='mpeg4', options=None)

if __name__=='__main__':
    #create_fla_data()

    #full_saved_path = '../saved_data/fla/processed_3tests_fla_model_outdoor_A_sym_01-21-2020-15-45-02.pt'
    #create_table_stats(uncertainty_metric_fn=sum_bingham_dispersion_coeff, data_file=full_saved_path)
    
    #full_saved_path = '../saved_data/fla/processed_fla_model_indoor_A_sym_01-21-2020-15-54-30.pt'
    #full_saved_path = '../saved_data/fla/processed_fla_model_outdoor_A_sym_01-21-2020-15-45-02.pt'
    #create_bar_and_scatter_plots(uncertainty_metric_fn=sum_bingham_dispersion_coeff, quantile=0.5, data_file=full_saved_path)
    # full_data_file = 'saved_data/fla/processed_video_fla_model_outdoor_A_sym_01-21-2020-15-45-02.pt'
    # create_video(full_data_file)

    create_fla_autoencoder_data()