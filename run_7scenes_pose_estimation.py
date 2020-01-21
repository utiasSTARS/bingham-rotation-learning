import torch
import time, argparse
from datetime import datetime
import numpy as np
from loaders import SevenScenesData
from networks import *
from losses import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tqdm
from helpers_train_test import train_test_model


def main():


    parser = argparse.ArgumentParser(description='7Scenes experiment')
    parser.add_argument('--scene', type=str, default='chess')
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

    parser.add_argument('--model', choices=['A_sym', '6D', 'quat'], default='A_sym')
    parser.add_argument('--lr', type=float, default=5e-4)

    args = parser.parse_args()
    print(args)

    #Float or Double?
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float


    #Load datasets
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.Rand(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

        #Monolith
    if args.megalith:
        dataset_dir = '/media/datasets/'
    else:
        dataset_dir = '/media/m2-drive/datasets/'

    data_folder = dataset_dir+'7scenes'

    train_loader = DataLoader(SevenScenesData(args.scene, data_folder, train=True, transform=transform_train, output_first_image=False, tensor_type=tensor_type),
                        batch_size=args.batch_size_train, pin_memory=True,
                        shuffle=True, num_workers=args.num_workers, drop_last=False)
    valid_loader = DataLoader(SevenScenesData(args.scene, data_folder, train=False, transform=transform_test, output_first_image=False, tensor_type=tensor_type),
                        batch_size=args.batch_size_test, pin_memory=True,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    
    dim_in = 3

    if args.model == 'A_sym':
        print('==============Using A (Sym) MODEL====================')
        model = QuatFlowNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob, dim_in=dim_in, batchnorm=args.batchnorm).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_chordal_squared_loss
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
        loss_fn = quat_chordal_squared_loss
        (train_stats, test_stats) = train_test_model(args, loss_fn, model, train_loader, valid_loader, tensorboard_output=False)


    if args.save_model:
        saved_data_file_name = '7scenes_model_{}_{}_{}'.format(args.model, args.scene, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/7scenes/{}.pt'.format(saved_data_file_name)
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
