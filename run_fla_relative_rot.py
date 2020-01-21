import torch
import time, argparse
from datetime import datetime
import numpy as np
from loaders import FLADataset
from networks import *
from losses import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tqdm
from helpers_train_test import train_test_model



def main():
    parser = argparse.ArgumentParser(description='KITTI relative odometry experiment')
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--batch_size_test', type=int, default=64)
    parser.add_argument('--batch_size_train', type=int, default=32)

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--megalith', action='store_true', default=False)

    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--optical_flow', action='store_true', default=False)
    parser.add_argument('--batchnorm', action='store_true', default=False)
    
    parser.add_argument('--unit_frob', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--enforce_psd', action='store_true', default=False)
    parser.add_argument('--scene', choices=['indoor', 'outdoor'], default='indoor')

    parser.add_argument('--model', choices=['A_sym', '6D', 'quat'], default='A_sym')
    parser.add_argument('--lr', type=float, default=5e-4)


    args = parser.parse_args()
    print(args)

    #Float or Double?
    tensor_type = torch.float


    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float



    #Monolith
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

    test_dataset = 'FLA/{}_test.csv'.format(args.scene)
    train_dataset = 'FLA/{}_train.csv'.format(args.scene)

    train_loader = DataLoader(FLADataset(train_dataset, image_dir=image_dir, pose_dir=pose_dir, transform=transform),
                            batch_size=args.batch_size_train, pin_memory=False,
                            shuffle=True, num_workers=args.num_workers, drop_last=False)

    valid_loader = DataLoader(FLADataset(test_dataset, image_dir=image_dir, pose_dir=pose_dir, transform=transform, eval_mode=True),
                            batch_size=args.batch_size_test, pin_memory=False,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)

    
    if args.model == 'A_sym':
        print('==============Using A (Sym) MODEL====================')
        model = QuatFlowNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_chordal_squared_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False, scheduler=False)

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
        loss_fn = quat_chordal_squared_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)

    if args.save_model:
        saved_data_file_name = 'fla_model_{}_{}_{}'.format(args.scene, args.model, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/fla/{}.pt'.format(saved_data_file_name)
        torch.save({
                'model_type': args.model,
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'model': model.state_dict(),
                'train_stats_rep': train_stats.detach().cpu(),
                'test_stats_rep': test_stats.detach().cpu(),
                'args': args,
            }, full_saved_path)

        print('Saved data to {}.'.format(full_saved_path))

if __name__=='__main__':
    main()
