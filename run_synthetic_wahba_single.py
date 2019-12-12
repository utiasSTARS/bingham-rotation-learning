import torch
import numpy as np
from networks import *
from quaternions import *
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
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--bidirectional_loss', action='store_true', default=False)
    parser.add_argument('--static_data', action='store_true', default=False)
    
    parser.add_argument('--pretrain_A_net', action='store_true', default=False)
    parser.add_argument('--use_A_prior', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--comparison', action='store_true', default=False)


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
        print('===================TRAINING DIRECT MODEL=======================')
        model_direct = QuatNetDirect(num_pts=args.matches_per_sample, bidirectional=args.bidirectional_loss).to(device=device, dtype=tensor_type)
        (train_stats_direct, test_stats_direct) = train_test_model(args, train_data, test_data, model_direct, tensorboard_output=True)

    #Train and test with new representation
    print('===================TRAINING REP MODEL=======================')
    A_net = ANet(num_pts=args.matches_per_sample, bidirectional=args.bidirectional_loss).to(device=device, dtype=tensor_type)
    model_rep = QuatNet(A_net=A_net).to(device=device, dtype=tensor_type)
    (train_stats_rep, test_stats_rep) = train_test_model(args, train_data, test_data, model_rep, tensorboard_output=True)

    
    if args.comparison:
        saved_data_file_name = 'synthetic_wahba_experiment_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = 'saved_data/synthetic/{}.pt'.format(saved_data_file_name)
        torch.save({
                'model_rep': model_rep.state_dict(),
                'model_direct': model_direct.state_dict(),
                'train_stats_direct': train_stats_direct.detach().cpu(),
                'test_stats_direct': test_stats_direct.detach().cpu(),
                'train_stats_rep': train_stats_rep.detach().cpu(),
                'test_stats_rep': test_stats_rep.detach().cpu(),
                'args': args,
            }, full_saved_path)

        print('Saved data to {}.'.format(full_saved_path))

if __name__=='__main__':
    main()
