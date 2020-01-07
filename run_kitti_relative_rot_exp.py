import torch
import time, argparse
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter
from loaders import KITTIVODatasetPreTransformed
from networks import *
from losses import *
from torch.utils.data import Dataset, DataLoader
from quaternions import *
import tqdm
from utils import loguniform
from helpers_train_test import train_test_model

def main():


    parser = argparse.ArgumentParser(description='KITTI relative odometry experiment')
    parser.add_argument('--epochs', type=int, default=25)

    parser.add_argument('--batch_size_test', type=int, default=64)
    parser.add_argument('--batch_size_train', type=int, default=32)

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--megalith', action='store_true', default=False)
    parser.add_argument('--batchnorm', action='store_true', default=False)

    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--optical_flow', action='store_true', default=False)
    
    #Randomly select within this range
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument('--trials', type=int, default=5)


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


    for seq in ['00','02','05']:
        print('===================SEQ {}======================='.format(seq))
        kitti_data_pickle_file = 'kitti/kitti_singlefile_data_sequence_{}_delta_2_reverse_True_min_turn_1.0.pickle'.format(seq)
        train_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='train', seq_prefix=seq_prefix),
                                batch_size=args.batch_size_train, pin_memory=False,
                                shuffle=True, num_workers=args.num_workers, drop_last=True)

        valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=args.optical_flow, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test', seq_prefix=seq_prefix),
                                batch_size=args.batch_size_test, pin_memory=False,
                                shuffle=False, num_workers=args.num_workers, drop_last=False)    
        train_stats_list = []
        test_stats_list = []

        lrs = torch.empty(args.trials)
        for t_i in range(args.trials):
            #Train and test direct model
            print('===================TRIAL {}/{}======================='.format(t_i+1, args.trials))

            lr = loguniform(np.log(args.lr_min), np.log(args.lr_max))
            args.lr = lr
            print('Learning rate: {:.3E}'.format(lr))

            #Train and test with new representation
            dim_in = 2 if args.optical_flow else 6

            print('==============TRAINING A (Sym) MODEL====================')
            model_sym = QuatFlowNet(enforce_psd=False, unit_frob_norm=args.batchnorm, dim_in=dim_in).to(device=device, dtype=tensor_type)
            train_loader.dataset.rotmat_targets = False
            valid_loader.dataset.rotmat_targets = False
            loss_fn = quat_squared_loss
            (train_stats_A_sym, test_stats_A_sym) = train_test_model(args, loss_fn, model_sym, train_loader, valid_loader, tensorboard_output=False)


            print('==========TRAINING DIRECT 6D ROTMAT MODEL============')
            model_6D = RotMat6DFlowNet(dim_in=dim_in).to(device=device, dtype=tensor_type)
            train_loader.dataset.rotmat_targets = True
            valid_loader.dataset.rotmat_targets = True
            loss_fn = rotmat_frob_squared_norm_loss
            (train_stats_6D, test_stats_6D) = train_test_model(args, loss_fn, model_6D, train_loader, valid_loader, tensorboard_output=False)

            print('=========TRAINING DIRECT QUAT MODEL==================')
            model_quat = BasicCNN(dim_in=dim_in, dim_out=4, normalize_output=True).to(device=device, dtype=tensor_type)
            train_loader.dataset.rotmat_targets = False
            valid_loader.dataset.rotmat_targets = False
            loss_fn = quat_squared_loss
            (train_stats_quat, test_stats_quat) = train_test_model(args, loss_fn, model_quat, train_loader, valid_loader, tensorboard_output=False)

                      # #Train and test with new representation
            # print('==============TRAINING A (PSD) MODEL====================')
            # model_psd = QuatNet(enforce_psd=True, unit_frob_norm=True).to(device=device, dtype=tensor_type)
            # loss_fn = quat_squared_loss
            # (train_stats_A_psd, test_stats_A_psd) = train_test_model(args, loss_fn, model_psd, train_loader, valid_loader, tensorboard_output=False)

            lrs[t_i] = lr
            train_stats_list.append([train_stats_6D, train_stats_quat, train_stats_A_sym])
            test_stats_list.append([test_stats_6D, test_stats_quat, test_stats_A_sym])
            
        saved_data_file_name = 'diff_lr_kitti_experiment_3models_seq_{}_{}'.format(seq, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/kitti/{}.pt'.format(saved_data_file_name)

        torch.save({
            'train_stats_list': train_stats_list,
            'test_stats_list': test_stats_list,
            'named_approaches': ['6D', 'Quat', 'A (sym)'],
            'learning_rates': lrs,
            'args': args
        }, full_saved_path)

        print('Saved data to {}.'.format(full_saved_path))

if __name__=='__main__':
    main()
