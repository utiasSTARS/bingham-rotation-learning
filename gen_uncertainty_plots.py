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


def evaluate_rotmat_model(loader, model):
    q_est = []
    q_target = []
    
    with torch.no_grad():
        model.eval()
        print('Loading rotmat model...')
        for _, (x, target) in enumerate(loader):
            #Move all data to appropriate device
            x = x.to(device=device, dtype=tensor_type)
            q = rotmat_to_quat(model.forward(x).squeeze().cpu())
            q_est.append(q)
            q_target.append(target.cpu())
            
    q_est = torch.cat(q_est, dim=0)
    q_target = torch.cat(q_target, dim=0)
    
    return (q_est, q_target)

def evaluate_A_model(loader, model):
    q_est = []
    q_target = []
    A_pred = []

    with torch.no_grad():
        model.eval()
        print('Loading A model...')
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
        A_predt, q_estt, q_targett = evaluate_A_model(train_loader, model)
        A_pred, q_est, q_target = evaluate_A_model(valid_loader, model)
        return ((A_predt, q_estt, q_targett), (A_pred, q_est, q_target))
    else:
        model = RotMat6DFlowNet(dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        model.load_state_dict(checkpoint['model'], strict=False)
        q_estt, q_targett = evaluate_rotmat_model(train_loader, model)
        q_est, q_target = evaluate_rotmat_model(valid_loader, model)
        return ((q_estt, q_targett), (q_est, q_target))

def create_kitti_data():

    prefix = 'saved_data/kitti/'
    file_list_6D = ['kitti_model_6D_seq_00_01-02-2020-14-21-01.pt', 'kitti_model_6D_seq_02_01-02-2020-15-13-10.pt','kitti_model_6D_seq_05_01-02-2020-16-09-34.pt']
    file_list_A_sym = ['kitti_model_A_sym_seq_00_01-01-2020-23-16-53.pt', 'kitti_model_A_sym_seq_02_01-02-2020-00-24-03.pt', 'kitti_model_A_sym_seq_05_01-01-2020-21-52-03.pt']
    
    print('Collecting normal data....')
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
                'data_6D': data_6D,
                'data_A': data_A,
                'data_6D_transformed': data_6D_transformed,
                'data_A_transformed': data_A_transformed,
                'transform_erase_prob': transform_erase_prob
    }, full_saved_path)

    print('Saved data to {}.'.format(full_saved_path))

    return

def create_plots():

    mean_err = quat_angle_diff(q_est, q_target, reduce=True)

    quantile = 0.75
    thresh = ll_threshold(A_predt, quantile)
    mask = wigner_log_likelihood(A_pred) < thresh

    mean_err_filter = quat_angle_diff(q_est[mask], q_target[mask])

if __name__=='__main__':
    create_kitti_data()
