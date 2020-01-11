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
from helpers_train_test import train_test_model

def main():


    parser = argparse.ArgumentParser(description='ShapeNet experiment')
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--batch_size_test', type=int, default=1)
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--rotations_per_batch_train', type=int, default=10)
    parser.add_argument('--rotations_per_batch_test', type=int, default=100)
    parser.add_argument('--iterations_per_epoch', type=int, default=200)

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batchnorm', action='store_true', default=False)

    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)

    
    parser.add_argument('--enforce_psd', action='store_true', default=False)
    parser.add_argument('--unit_frob', action='store_true', default=False)

    parser.add_argument('--model', choices=['A_sym', '6D', 'quat'], default='A_sym')

    parser.add_argument('--lr', type=float, default=5e-4)



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

    if args.model == 'A_sym':
        print('==============TRAINING A (Sym) MODEL====================')
        model = QuatNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob,batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_squared_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)

    elif args.model == '6D':
        print('==========TRAINING DIRECT 6D ROTMAT MODEL============')
        model = RotMat6DDirect(batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = True
        valid_loader.dataset.rotmat_targets = True
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)

    elif args.model == 'quat':

        print('=========TRAINING DIRECT QUAT MODEL==================')
        model = PointNet(dim_out=4, normalize_output=True, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_squared_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)
        
    if args.save_model:
        saved_data_file_name = 'shapenet_model_{}_{}'.format(args.model, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/shapenet/{}.pt'.format(saved_data_file_name)
        torch.save({
                'model_type': args.model,
                'model': model.state_dict(),
                'train_stats': train_stats.detach().cpu(),
                'test_stats': test_stats.detach().cpu(),
                'args': args,
            }, full_saved_path)

        print('Saved data to {}.'.format(full_saved_path))


if __name__=='__main__':
    main()
