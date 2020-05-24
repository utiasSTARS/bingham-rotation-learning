import torch
import numpy as np
from networks import *
from losses import *
from helpers_sim import *
import argparse


def main():
    parser = argparse.ArgumentParser(description='Synthetic Wahba arguments.')
    parser.add_argument('--sim_sigma', type=float, default=1e-2)
    parser.add_argument('--N_train', type=int, default=500)
    parser.add_argument('--N_test', type=int, default=100)
    parser.add_argument('--matches_per_sample', type=int, default=25)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size_train', type=int, default=100)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--dataset', choices=['static', 'dynamic', 'dynamic_beachball'], default='dynamic')
    parser.add_argument('--beachball_sigma_factors', type=lambda s: [float(item) for item in s.split(',')], default=[0.1, 0.5, 2, 10], help='Heteroscedastic point cloud that has different noise levels (resembling a beachball).')
    parser.add_argument('--max_rotation_angle', type=float, default=180., help='In degrees. Maximum axis-angle rotation of simulated rotation.')

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--enforce_psd', action='store_true', default=False)
    parser.add_argument('--unit_frob', action='store_true', default=False)

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float


    #Generate data
    if args.dataset == 'static':
        train_data, test_data = create_experimental_data_fast(args.N_train, args.N_test, args.matches_per_sample, sigma=args.sim_sigma, device=device, dtype=tensor_type)
    else:
        #Data will be generated on the fly
        train_data, test_data = None, None


    print('===================TRAINING UNIT QUATERNION MODEL=======================')
    model = PointNet(dim_out=4, normalize_output=True).to(device=device, dtype=tensor_type)
    #loss_fn = quat_squared_loss
    loss_fn = quat_chordal_squared_loss
    (_, _) = train_test_model(args, train_data, test_data, model, loss_fn, rotmat_targets=False, tensorboard_output=True)

    #Train and test direct model
    print('===================TRAINING 6D ROTMAT MODEL=======================')
    model = RotMat6DDirect().to(device=device, dtype=tensor_type)
    loss_fn = rotmat_frob_squared_norm_loss
    (_, _) = train_test_model(args, train_data, test_data, model, loss_fn,  rotmat_targets=True, tensorboard_output=True)

    print('===================TRAINING A SYM MODEL=======================')
    model = QuatNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob).to(device=device, dtype=tensor_type)
    #loss_fn = quat_squared_loss
    loss_fn = quat_chordal_squared_loss
    (train_stats, test_stats) = train_test_model(args, train_data, test_data, model, loss_fn,  rotmat_targets=False, tensorboard_output=True)


if __name__=='__main__':
    main()
