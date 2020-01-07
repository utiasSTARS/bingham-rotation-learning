import torch
import time, datetime, argparse
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime
from loaders import SevenScenesData
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from quaternions import *
from networks import *
import tqdm


def main():


    parser = argparse.ArgumentParser(description='7Scenes experiment')
    parser.add_argument('--scene', type=str, default='chess')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_test', type=int, default=32)
    parser.add_argument('--comparison', action='store_true', default=False)
    parser.add_argument('--dual_network', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    

    args = parser.parse_args()
    print(args)

    #Float or Double?
    tensor_type = torch.float


    #Load datasets
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
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

    if not args.cuda:
        data_folder = '/Users/valentinp/Desktop/datasets/7scenes'
        device = torch.device('cpu')
    else:
        data_folder = '/media/m2-drive/datasets/7scenes'
        device = torch.device('cuda:0')

    train_loader = DataLoader(SevenScenesData(args.scene, data_folder, train=True, transform=transform_train, output_first_image=args.dual_network, tensor_type=tensor_type),
                        batch_size=args.batch_size_train, pin_memory=True,
                        shuffle=True, num_workers=args.num_workers, drop_last=False)
    valid_loader = DataLoader(SevenScenesData(args.scene, data_folder, train=False, transform=transform_test, output_first_image=args.dual_network, tensor_type=tensor_type),
                        batch_size=args.batch_size_test, pin_memory=True,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    

    #Train and test with new representation
    print('===================TRAINING REP MODEL=======================')
    model_rep = CustomResNetConvex(dual=args.dual_network)
    model_rep.to(dtype=tensor_type, device=device)
    loss_fn = quat_squared_loss
    (train_stats_rep, test_stats_rep) = train_test_model(args, loss_fn, model_rep, train_loader, valid_loader)


    if args.comparison:
        #Train and test direct model
        print('===================TRAINING DIRECT MODEL=======================')

        model_direct = CustomResNetDirect(dual=args.dual_network)
        model_direct.to(dtype=tensor_type, device=device)
        loss_fn = quat_squared_loss
        (train_stats_direct, test_stats_direct) = train_test_model(args, loss_fn, model_direct, train_loader, valid_loader)

    if args.comparison: 
        saved_data_file_name = '7scenes_experiment_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/7scenes/{}.pt'.format(saved_data_file_name)
        torch.save({
                # 'model_rep': model_rep.state_dict(),
                # 'model_direct': model_direct.state_dict(),
                'train_stats_direct': train_stats_direct.detach().cpu(),
                'test_stats_direct': test_stats_direct.detach().cpu(),
                'train_stats_rep': train_stats_rep.detach().cpu(),
                'test_stats_rep': test_stats_rep.detach().cpu(),
                'args': args,
            }, full_saved_path)

        print('Saved data to {}.'.format(full_saved_path))

if __name__=='__main__':
    main()
