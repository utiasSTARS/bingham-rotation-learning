import torch
import numpy as np
from networks import *
from losses import *
from helpers_sim import *
from datetime import datetime
import argparse


def main():


    parser = argparse.ArgumentParser(description='Synthetic Wahba arguments.')
    parser.add_argument('--sim_sigma', type=float, default=1e-2)
    parser.add_argument('--N_train', type=int, default=500)
    parser.add_argument('--N_test', type=int, default=100)
    parser.add_argument('--matches_per_sample', type=int, default=100)

    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size_train', type=int, default=100)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--dataset', choices=['static', 'dynamic', 'dynamic_beachball'], default='dynamic')
    parser.add_argument('--max_rotation_angle', type=float, default=180., help='In degrees. Maximum axis-angle rotation of simulated rotation.')
    parser.add_argument('--beachball_sigma_factors', type=lambda s: [float(item) for item in s.split(',')], default=[0.1, 0.5, 2, 10])

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--enforce_psd', action='store_true', default=False)
    parser.add_argument('--unit_frob', action='store_true', default=False)

    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--model', choices=['A_sym', 'A_sym_rot', 'A_sym_rot_16', '6D', 'quat'], default='A_sym')


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


    if args.model == '6D':
        #Train and test direct model
        print('===================TRAINING DIRECT 6D ROTMAT MODEL=======================')
        model = RotMat6DDirect().to(device=device, dtype=tensor_type)
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats, test_stats) = train_test_model(args, train_data, test_data, model, loss_fn,  rotmat_targets=True, tensorboard_output=True)

    elif args.model == 'quat':
        print('===================TRAINING DIRECT QUAT MODEL=======================')
        model = PointNet(dim_out=4, normalize_output=True).to(device=device, dtype=tensor_type)
        #loss_fn = quat_squared_loss
        loss_fn = quat_chordal_squared_loss
        (_, _) = train_test_model(args, train_data, test_data, model, loss_fn, rotmat_targets=False, tensorboard_output=True)
    
    #Train and test with new representation
    elif args.model == 'A_sym':
        print('===================TRAINING A sym (Quat) MODEL=======================')
        model = QuatNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob).to(device=device, dtype=tensor_type)
        #loss_fn = quat_squared_loss
        loss_fn = quat_chordal_squared_loss
        (train_stats, test_stats) = train_test_model(args, train_data, test_data, model, loss_fn,  rotmat_targets=False, tensorboard_output=True)


    #Train and test with new representation
    elif args.model == 'A_sym_rot':
        print('===================TRAINING A sym (55 param RotMat) MODEL=======================')
        model = RotMatSDPNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob).to(device=device, dtype=tensor_type)
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats, test_stats) = train_test_model(args, train_data, test_data, model, loss_fn,  rotmat_targets=True, tensorboard_output=True)

    #Train and test with new representation
    elif args.model == 'A_sym_rot_16':
        print('===================TRAINING A sym (16 param RotMat) MODEL=======================')
        model = RotMatSDPNet(dim_rep=16, enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob).to(device=device, dtype=tensor_type)
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats, test_stats) = train_test_model(args, train_data, test_data, model, loss_fn,  rotmat_targets=True, tensorboard_output=True)


    if args.save_model:
        saved_data_file_name = 'synthetic_wahba_model_{}_{}'.format(args.model, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/synthetic/{}.pt'.format(saved_data_file_name)
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
