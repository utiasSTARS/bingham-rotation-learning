import torch
import numpy as np
from networks import *
from quaternions import *
from sim_helpers import *
from datetime import datetime
import argparse


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))

def main():
    parser = argparse.ArgumentParser(description='Synthetic Wahba arguments.')
    parser.add_argument('--sim_sigma', type=float, default=1e-6)
    parser.add_argument('--N_train', type=int, default=500)
    parser.add_argument('--N_test', type=int, default=100)
    parser.add_argument('--matches_per_sample', type=int, default=1000)

    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size_train', type=int, default=100)
    parser.add_argument('--batch_size_test', type=int, default=100)
    #parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--bidirectional_loss', action='store_true', default=False)
    parser.add_argument('--static_data', action='store_true', default=False)
    
    parser.add_argument('--pretrain_A_net', action='store_true', default=False)
    parser.add_argument('--use_A_prior', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--comparison', action='store_true', default=False)


    #Randomly select within this range
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument('--trials', type=int, default=25)
    

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

    stats_list = []

    for t_i in range(args.trials):
        #Train and test direct model
        print('===================TRIAL {}/{}======================='.format(t_i+1, args.trials))

        lr = loguniform(np.log(args.lr_min), np.log(args.lr_max))
        args.lr = lr
        print('Learning rate: {:.3E}'.format(lr))

        print('====DIRECT MODEL====')
        model_direct = QuatNetDirect(num_pts=args.matches_per_sample, bidirectional=args.bidirectional_loss).to(device=device, dtype=tensor_type)
        (train_stats_direct, test_stats_direct) = train_test_model(args, train_data, test_data, model_direct, tensorboard_output=False)

        #Train and test with new representation
        print('====REP MODEL====')
        A_net = ANet(num_pts=args.matches_per_sample, bidirectional=args.bidirectional_loss).to(device=device, dtype=tensor_type)
        model_rep = QuatNet(A_net=A_net).to(device=device, dtype=tensor_type)
        (train_stats_rep, test_stats_rep) = train_test_model(args, train_data, test_data, model_rep, tensorboard_output=False)

        stats_list.append([lr, train_stats_direct, train_stats_rep, test_stats_direct, test_stats_rep])
    

    saved_data_file_name = 'diff_lr_synthetic_wahba_experiment_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = 'saved_data/synthetic/{}.pt'.format(saved_data_file_name)

    torch.save({
        'stats_list': stats_list,
        'args': args
    }, full_saved_path)
    print('Saved data to {}.'.format(full_saved_path))

    
if __name__=='__main__':
    main()
