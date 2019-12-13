import torch
import numpy as np
from networks import *
from losses import *
from sim_helpers import *
from datetime import datetime
import argparse


def main():


    parser = argparse.ArgumentParser(description='Synthetic Wahba arguments.')
    parser.add_argument('--sim_sigma', type=float, default=1e-6)
    parser.add_argument('--N_train', type=int, default=500)
    parser.add_argument('--N_test', type=int, default=100)
    parser.add_argument('--matches_per_sample', type=int, default=1000)

    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size_train', type=int, default=100)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--static_data', action='store_true', default=False)
    
    parser.add_argument('--pretrain_A_net', action='store_true', default=False)
    parser.add_argument('--use_A_prior', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--comparison', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)


    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float


    #Generate data
    if args.static_data:
        train_data, test_data = create_experimental_data_fast(args.N_train, args.N_test, args.matches_per_sample, sigma=args.sim_sigma, device=device, dtype=tensor_type)
    else:
        #Data will be generated on the fly
        train_data, test_data = None, None

    #Train and test direct model
    if args.comparison:
        print('===================TRAINING DIRECT 6D ROTMAT MODEL=======================')
        model_6D = RotMat6DDirect().to(device=device, dtype=tensor_type)
        loss_fn = rotmat_frob_squared_norm_loss
        (_, _) = train_test_model(args, train_data, test_data, model_6D, loss_fn, rotmat_targets=True, tensorboard_output=True)


        print('===================TRAINING DIRECT QUAT MODEL=======================')
        model_quat = PointNet(dim_out=4, normalize_output=True).to(device=device, dtype=tensor_type)
        loss_fn = quat_squared_loss
        (_, _) = train_test_model(args, train_data, test_data, model_quat, loss_fn, rotmat_targets=False, tensorboard_output=True)

    #Train and test with new representation
    print('===================TRAINING REP MODEL=======================')
    model_rep = QuatNet().to(device=device, dtype=tensor_type)
    loss_fn = quat_squared_loss
    (train_stats_rep, test_stats_rep) = train_test_model(args, train_data, test_data, model_rep, loss_fn,  rotmat_targets=False, tensorboard_output=True)

    
    if args.save_model:
        saved_data_file_name = 'synthetic_wahba_model_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/synthetic/{}.pt'.format(saved_data_file_name)
        torch.save({
                'model_rep': model_rep.state_dict(),
                'train_stats_rep': train_stats_rep.detach().cpu(),
                'test_stats_rep': test_stats_rep.detach().cpu(),
                'args': args,
            }, full_saved_path)

        print('Saved data to {}.'.format(full_saved_path))

if __name__=='__main__':
    main()
