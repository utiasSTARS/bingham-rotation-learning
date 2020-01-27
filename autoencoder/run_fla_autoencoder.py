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


if __name__=='__main__':
    main()
