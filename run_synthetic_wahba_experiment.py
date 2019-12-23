import torch
import numpy as np
from networks import *
from quaternions import *
from helpers_sim import *
from datetime import datetime
import argparse
from utils import loguniform

def main():
    parser = argparse.ArgumentParser(description='Synthetic Wahba arguments.')
    parser.add_argument('--sim_sigma', type=float, default=1e-2)
    parser.add_argument('--beachball_sigma_factors', type=lambda s: [float(item) for item in s.split(',')], default=[0.1, 0.5, 2, 10])

    parser.add_argument('--N_train', type=int, default=500)
    parser.add_argument('--N_test', type=int, default=100)
    parser.add_argument('--matches_per_sample', type=int, default=100)

    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size_train', type=int, default=100)
    parser.add_argument('--batch_size_test', type=int, default=100)
    #parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--dataset', choices=['static', 'dynamic', 'dynamic_beachball'], default='dynamic')
    
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)

    #Randomly select within this range
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument('--trials', type=int, default=25)
    

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

    train_stats_list = []
    test_stats_list = []
    lrs = torch.empty(args.trials)
    for t_i in range(args.trials):
        #Train and test direct model
        print('===================TRIAL {}/{}======================='.format(t_i+1, args.trials))

        lr = loguniform(np.log(args.lr_min), np.log(args.lr_max))
        args.lr = lr
        print('Learning rate: {:.3E}'.format(lr))

        print('==========TRAINING DIRECT 6D ROTMAT MODEL============')
        model_6D = RotMat6DDirect().to(device=device, dtype=tensor_type)
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats_6d, test_stats_6d) = train_test_model(args, train_data, test_data, model_6D, loss_fn, rotmat_targets=True, tensorboard_output=False)


        print('=========TRAINING DIRECT QUAT MODEL==================')
        model_quat = PointNet(dim_out=4, normalize_output=True).to(device=device, dtype=tensor_type)
        loss_fn = quat_squared_loss
        (train_stats_quat, test_stats_quat) = train_test_model(args, train_data, test_data, model_quat, loss_fn, rotmat_targets=False, tensorboard_output=False)

        #Train and test with new representation
        print('==============TRAINING A (16 sym quat) MODEL====================')
        model_A_sym = QuatNet(enforce_psd=False, unit_frob_norm=True).to(device=device, dtype=tensor_type)
        loss_fn = quat_squared_loss
        (train_stats_A_sym, test_stats_A_sym) = train_test_model(args, train_data, test_data, model_A_sym, loss_fn,  rotmat_targets=False, tensorboard_output=False)

        #Train and test with new representation
        print('==============TRAINING A (16 psd quat) MODEL====================')
        model_A_psd = QuatNet(enforce_psd=True, unit_frob_norm=True).to(device=device, dtype=tensor_type)
        loss_fn = quat_squared_loss
        (train_stats_A_psd, test_stats_A_psd) = train_test_model(args, train_data, test_data, model_A_psd, loss_fn,  rotmat_targets=False, tensorboard_output=False)

        print('==============TRAINING A (55 psd rotmat) MODEL====================')
        model_A_rotmat = RotMatSDPNet(enforce_psd=False, unit_frob_norm=True).to(device=device, dtype=tensor_type)
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats_A_rotmat, test_stats_A_rotmat) = train_test_model(args, train_data, test_data, model_A_rotmat, loss_fn,  rotmat_targets=True, tensorboard_output=False)

        #Memory leak in cvxpylayer?
        del(model_A_rotmat)
        
        lrs[t_i] = lr
        train_stats_list.append([train_stats_6d, train_stats_quat, train_stats_A_sym, train_stats_A_psd, train_stats_A_rotmat])
        test_stats_list.append([test_stats_6d, test_stats_quat, test_stats_A_sym, test_stats_A_psd, test_stats_A_rotmat])
        
    saved_data_file_name = 'diff_lr_synthetic_wahba_experiment_5models_{}_{}'.format(args.dataset, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = 'saved_data/synthetic/{}.pt'.format(saved_data_file_name)

    torch.save({
        'train_stats_list': train_stats_list,
        'test_stats_list': test_stats_list,
        'named_approaches': ['6D', 'Quat', 'A (quat-sym)', 'A (quat-psd)', 'A (rotmat-sym)'],
        'learning_rates': lrs,
        'args': args
    }, full_saved_path)

    print('Saved data to {}.'.format(full_saved_path))

    
if __name__=='__main__':
    main()