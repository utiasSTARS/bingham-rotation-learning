import torch
import time, argparse
from datetime import datetime
import numpy as np
from loaders import KITTIVODatasetPreTransformed
from networks import *
from losses import *
from torch.utils.data import Dataset, DataLoader
import tqdm
from helpers_train_test import train_test_model



def main():
    parser = argparse.ArgumentParser(description='KITTI relative odometry experiment')
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--batch_size_test', type=int, default=64)
    parser.add_argument('--batch_size_train', type=int, default=32)

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--megalith', action='store_true', default=False)

    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--optical_flow', action='store_true', default=False)
    parser.add_argument('--batchnorm', action='store_true', default=False)
    
    parser.add_argument('--unit_frob', action='store_true', default=False)

    parser.add_argument('--seq', choices=['00', '02', '05'], default='00')
    parser.add_argument('--model', choices=['A_sym', '6D', 'quat'], default='A_sym')

    #Randomly select within this range
    parser.add_argument('--lr', type=float, default=5e-4)


    args = parser.parse_args()
    print(args)

    #Float or Double?
    tensor_type = torch.float


    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float

    transform = None
    seqs_base_path = '/media/m2-drive/datasets/KITTI/single_files'
    if args.megalith:
        seqs_base_path = '/media/datasets/KITTI/single_files'

    seq_prefix = 'seq_'

    #kitti_data_pickle_file = 'kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_True_minta_0.0.pickle'.format(args.seq)
    kitti_data_pickle_file = 'kitti/kitti_singlefile_data_sequence_{}_delta_2_reverse_True_min_turn_2.0.pickle'.format(args.seq)
    
    train_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='train', seq_prefix=seq_prefix),
                            batch_size=args.batch_size_train, pin_memory=False,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)

    valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test', seq_prefix=seq_prefix),
                            batch_size=args.batch_size_test, pin_memory=False,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)
    #Train and test with new representation
    dim_in = 2 if args.optical_flow else 6

    if args.model == 'A_sym':
        print('==============Using A (Sym) MODEL====================')
        model_sym = QuatFlowNet(enforce_psd=False, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_squared_loss
        (train_stats_A_sym, test_stats_A_sym) = train_test_model(args, loss_fn, model_sym, train_loader, valid_loader, tensorboard_output=False)

    elif args.model == '6D':
        print('==========TRAINING DIRECT 6D ROTMAT MODEL============')
        model_6D = RotMat6DFlowNet(dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = True
        valid_loader.dataset.rotmat_targets = True
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats_6D, test_stats_6D) = train_test_model(args, loss_fn, model_6D, train_loader, valid_loader, tensorboard_output=False)

    elif args.model == 'quat':
        print('=========TRAINING DIRECT QUAT MODEL==================')
        model_quat = BasicCNN(dim_in=dim_in, dim_out=4, normalize_output=True, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_squared_loss
        (train_stats_quat, test_stats_quat) = train_test_model(args, loss_fn, model_quat, train_loader, valid_loader, tensorboard_output=False)

if __name__=='__main__':
    main()
