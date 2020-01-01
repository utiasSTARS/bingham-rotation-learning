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
    parser.add_argument('--save_model', action='store_true', default=False)

    parser.add_argument('--seq', choices=['00', '02', '05'], default='00')
    parser.add_argument('--model', choices=['A_sym', 'A_sym_rot', 'A_sym_rot_16', '6D', 'quat'], default='A_sym')

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
    kitti_data_pickle_file = 'kitti/kitti_singlefile_data_sequence_{}_delta_2_reverse_True_min_turn_1.0.pickle'.format(args.seq)
    
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
        model = QuatFlowNet(enforce_psd=False, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_squared_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)

    elif args.model == 'A_sym_rot_16':
        print('==============Using A (Sym 16) RotMat MODEL====================')
        model = RotMatSDPFlowNet(dim_rep=16, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = True
        valid_loader.dataset.rotmat_targets = True
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)

    elif args.model == 'A_sym_rot':
        print('==============Using A (Sym) RotMat MODEL====================')
        model = RotMatSDPFlowNet(dim_rep=55, enforce_psd=True, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = True
        valid_loader.dataset.rotmat_targets = True
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)

    elif args.model == '6D':
        print('==========TRAINING DIRECT 6D ROTMAT MODEL============')
        model = RotMat6DFlowNet(dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = True
        valid_loader.dataset.rotmat_targets = True
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)

    elif args.model == 'quat':
        print('=========TRAINING DIRECT QUAT MODEL==================')
        model = BasicCNN(dim_in=dim_in, dim_out=4, normalize_output=True, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_squared_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)

    if args.save_model:
        saved_data_file_name = 'kitti_model_{}_seq_{}_{}'.format(args.model, args.seq, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/kitti/{}.pt'.format(saved_data_file_name)
        torch.save({
                'model_type': args.model,
                'model': model.state_dict(),
                'train_stats_rep': train_stats.detach().cpu(),
                'test_stats_rep': test_stats.detach().cpu(),
                'args': args,
            }, full_saved_path)

        print('Saved data to {}.'.format(full_saved_path))

if __name__=='__main__':
    main()
