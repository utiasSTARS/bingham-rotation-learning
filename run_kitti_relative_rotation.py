import torch
import time, argparse
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter
from loaders import KITTIVODatasetPreTransformed
from networks import *
from losses import *
from torch.utils.data import Dataset, DataLoader
from quaternions import *
import tqdm
from utils import loguniform



#Generic training function
def train(model, loss_fn, optimizer, x, q_gt):

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    q_est = model.forward(x)
    
    loss = loss_fn(q_est, q_gt)

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    return (q_est, loss.item())


def test(model, loss_fn, x, q_gt):
    # Forward
    with torch.no_grad():
        q_est = model.forward(x)
        loss = loss_fn(q_est, q_gt)
            
    return (q_est, loss.item())


def train_test_model(args, loss_fn, model, train_loader, test_loader, tensorboard_output=True):

    if tensorboard_output:
        writer = SummaryWriter()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)

    #Save stats
    train_stats = torch.zeros(args.epochs, 2)
    test_stats = torch.zeros(args.epochs, 2)
    
    device = next(model.parameters()).device

    pbar = tqdm.tqdm(total=args.epochs)
    for e in range(args.epochs):
        start_time = time.time()

        #Train model
        model.train()
        train_loss = torch.tensor(0.)
        train_mean_err = torch.tensor(0.)
        num_train_batches = len(train_loader)
        for batch_idx, (x, q_gt) in enumerate(train_loader):
            #Move all data to appropriate device
            q_gt = q_gt.to(device)
            x = x.to(device)
            (q_est, train_loss_k) = train(model, loss_fn, optimizer, x, q_gt)
            train_loss += (1./num_train_batches)*train_loss_k
            train_mean_err += (1./num_train_batches)*quat_angle_diff(q_est, q_gt)

        #Test model
        model.eval()
        num_test_batches = len(test_loader)
        test_loss = torch.tensor(0.)
        test_mean_err = torch.tensor(0.)

        for batch_idx, (x, q_gt) in enumerate(test_loader):
            #Move all data to appropriate device
            q_gt = q_gt.to(device)
            x = x.to(device)
            (q_est, test_loss_k) = test(model, loss_fn, x, q_gt)
            test_loss += (1./num_test_batches)*test_loss_k
            test_mean_err += (1./num_test_batches)*quat_angle_diff(q_est, q_gt)

        test_stats[e, 0] = test_loss
        test_stats[e, 1] = test_mean_err

        if tensorboard_output:
            writer.add_scalar('validation/loss', test_loss, e)
            writer.add_scalar('validation/mean_err', test_mean_err, e)

            writer.add_scalar('training/loss', train_loss, e)
            writer.add_scalar('training/mean_err', train_mean_err, e)
        
        
        #History tracking
        train_stats[e, 0] = train_loss
        train_stats[e, 1] = train_mean_err

        elapsed_time = time.time() - start_time
        
        output_string = 'Epoch: {}/{}. Train: Loss {:.3E} / Error {:.3f} (deg) | Test: Loss {:.3E} / Error {:.3f} (deg). Epoch time: {:.3f} sec.'.format(e+1, args.epochs, train_loss, train_mean_err, test_loss, test_mean_err, elapsed_time)
        pbar.set_description(output_string)
        pbar.update(1)


    if tensorboard_output:
        writer.close()
    pbar.close()
    return train_stats, test_stats

def main():


    parser = argparse.ArgumentParser(description='KITTI relative odometry experiment')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seq', type=str, default='00')

    parser.add_argument('--batch_size_test', type=int, default=16)
    parser.add_argument('--batch_size_train', type=int, default=16)

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--double', action='store_true', default=False)
    
    #Randomly select within this range
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument('--trials', type=int, default=25)


    args = parser.parse_args()
    print(args)

    #Float or Double?
    tensor_type = torch.float


    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float

    transform = None
    kitti_data_pickle_file = 'kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_True_minta_0.0.pickle'.format(args.seq)
    seqs_base_path = '/media/m2-drive/datasets/KITTI/single_files'
    seq_prefix = 'seq_'

    train_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, seqs_base_path=seqs_base_path, transform_img=transform, run_type='train', seq_prefix=seq_prefix),
                              batch_size=args.batch_size_train, pin_memory=False,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)

    valid_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, seqs_base_path=seqs_base_path, transform_img=transform, run_type='test', seq_prefix=seq_prefix),
                              batch_size=args.batch_size_test, pin_memory=False,
                              shuffle=False, num_workers=args.num_workers, drop_last=False)    
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
        model_6D = RotMat6DResNet().to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = True
        valid_loader.dataset.rotmat_targets = True
        loss_fn = rotmat_frob_squared_norm_loss
        (train_stats_6D, test_stats_6D) = train_test_model(args, loss_fn, model_6D, train_loader, valid_loader, tensorboard_output=False)


        print('=========TRAINING DIRECT QUAT MODEL==================')
        model_quat = CustomResNet(dim_out=4, normalize_output=True).to(device=device, dtype=tensor_type)
        train_loader.dataset.rotmat_targets = False
        valid_loader.dataset.rotmat_targets = False
        loss_fn = quat_squared_loss
        (train_stats_quat, test_stats_quat) = train_test_model(args, loss_fn, model_quat, train_loader, valid_loader, tensorboard_output=False)

        #Train and test with new representation
        print('==============TRAINING A (Sym) MODEL====================')
        model_sym = QuatResNet(enforce_psd=False, unit_frob_norm=True).to(device=device, dtype=tensor_type)
        loss_fn = quat_squared_loss
        (train_stats_A_sym, test_stats_A_sym) = train_test_model(args, loss_fn, model_sym, train_loader, valid_loader, tensorboard_output=False)

        # #Train and test with new representation
        # print('==============TRAINING A (PSD) MODEL====================')
        # model_psd = QuatNet(enforce_psd=True, unit_frob_norm=True).to(device=device, dtype=tensor_type)
        # loss_fn = quat_squared_loss
        # (train_stats_A_psd, test_stats_A_psd) = train_test_model(args, loss_fn, model_psd, train_loader, valid_loader, tensorboard_output=False)

        lrs[t_i] = lr
        # train_stats_list.append([train_stats_6D, train_stats_quat, train_stats_A_sym, train_stats_A_psd])
        # test_stats_list.append([test_stats_6D, test_stats_quat, test_stats_A_sym, test_stats_A_psd])
        
    # saved_data_file_name = 'diff_lr_shapenet_experiment_4models_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    # full_saved_path = 'saved_data/shapenet/{}.pt'.format(saved_data_file_name)

    # torch.save({
    #     'train_stats_list': train_stats_list,
    #     'test_stats_list': test_stats_list,
    #     'named_approaches': ['6D', 'Quat', 'A (sym)', 'A (psd)'],
    #     'learning_rates': lrs,
    #     'args': args
    # }, full_saved_path)

    # print('Saved data to {}.'.format(full_saved_path))

if __name__=='__main__':
    main()
