import torch
import time, argparse
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter
from loaders import PointNetDataset, pointnet_collate
from networks import *
from losses import *
from torch.utils.data import Dataset, DataLoader
from quaternions import *
import tqdm
from utils import loguniform
from train_test_helpers import train_test_model

def main():


    parser = argparse.ArgumentParser(description='ShapeNet experiment')
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--batch_size_test', type=int, default=1)
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--rotations_per_batch_train', type=int, default=10)
    parser.add_argument('--rotations_per_batch_test', type=int, default=100)
    parser.add_argument('--iterations_per_epoch', type=int, default=250)

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batchnorm', action='store_true', default=False)

    parser.add_argument('--double', action='store_true', default=False)
    
    #Randomly select within this range
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument('--trials', type=int, default=10)


    args = parser.parse_args()
    print(args)

    #Float or Double?
    tensor_type = torch.float


    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float

    if args.cuda:
        pointnet_data = '/home/valentinp/research/RotationContinuity/shapenet/data/pc_plane'
    else:
        pointnet_data = '/Users/valentinp/Dropbox/Postdoc/projects/misc/RotationContinuity/shapenet/data/pc_plane'
    
    train_loader = DataLoader(PointNetDataset(pointnet_data + '/points', load_into_memory=True, device=device, rotations_per_batch=args.rotations_per_batch_train, total_iters=args.iterations_per_epoch, dtype=tensor_type),
                        batch_size=args.batch_size_train, pin_memory=True, collate_fn=pointnet_collate,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)

    valid_loader = DataLoader(PointNetDataset(pointnet_data + '/points_test', load_into_memory=True, device=device, rotations_per_batch=args.rotations_per_batch_test, dtype=tensor_type, test_mode=True),
                        batch_size=args.batch_size_test, pin_memory=True, collate_fn=pointnet_collate,
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

        print('=========TRAINING DIRECT QUAT MODEL==================')
        model_quat = PointNet(dim_out=4, normalize_output=True, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_squared_loss
        (train_stats_quat, test_stats_quat) = train_test_model(args, loss_fn, model_quat, train_loader, valid_loader, tensorboard_output=False)

        print('==========TRAINING DIRECT 6D ROTMAT MODEL============')
        model_6D = RotMat6DDirect(batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = True
        valid_loader.dataset.rotmat_targets = True
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats_6D, test_stats_6D) = train_test_model(args, loss_fn, model_6D, train_loader, valid_loader, tensorboard_output=False)


               #Train and test with new representation
        print('==============TRAINING A (Sym) MODEL====================')
        model_sym = QuatNet(enforce_psd=False, unit_frob_norm=True,batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_squared_loss
        (train_stats_A_sym, test_stats_A_sym) = train_test_model(args, loss_fn, model_sym, train_loader, valid_loader, tensorboard_output=False)

        # #Train and test with new representation
        # print('==============TRAINING A (PSD) MODEL====================')
        # model_psd = QuatNet(enforce_psd=True, unit_frob_norm=True).to(device=device, dtype=tensor_type)
        # loss_fn = quat_squared_loss
        # (train_stats_A_psd, test_stats_A_psd) = train_test_model(args, loss_fn, model_psd, train_loader, valid_loader, tensorboard_output=False)

        lrs[t_i] = lr
        #train_stats_list.append([train_stats_6D, train_stats_quat, train_stats_A_sym, train_stats_A_psd])
        #test_stats_list.append([test_stats_6D, test_stats_quat, test_stats_A_sym, test_stats_A_psd])
        train_stats_list.append([train_stats_6D, train_stats_quat, train_stats_A_sym])
        test_stats_list.append([test_stats_6D, test_stats_quat, test_stats_A_sym])
        
    saved_data_file_name = 'diff_lr_shapenet_experiment_3models_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = 'saved_data/shapenet/{}.pt'.format(saved_data_file_name)

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
