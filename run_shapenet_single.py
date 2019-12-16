import torch
import time, datetime, argparse
import numpy as np
from tensorboardX import SummaryWriter
from loaders import PointNetDataset, pointnet_collate
from networks import *
from losses import *
from torch.utils.data import Dataset, DataLoader
from quaternions import *
import tqdm



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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)

    #Save stats
    train_stats = torch.empty(args.epochs, 2)
    test_stats = torch.empty(args.epochs, 2)
    
    device = next(model.parameters()).device

    
    for e in range(args.epochs):
        start_time = time.time()

        #Train model
        print('Epoch {} / {} | Training with lr: {:.3E}'.format(e+1, args.epochs, scheduler.get_lr()[0]))
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
        print('Testing...')
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


        scheduler.step()

        if tensorboard_output:
            writer.add_scalar('training/loss', train_loss, e)
            writer.add_scalar('training/mean_err', train_mean_err, e)

            writer.add_scalar('validation/loss', test_loss, e)
            writer.add_scalar('validation/mean_err', test_mean_err, e)
        
        #History tracking
        train_stats[e, 0] = train_loss
        train_stats[e, 1] = train_mean_err
        test_stats[e, 0] = test_loss
        test_stats[e, 1] = test_mean_err

        elapsed_time = time.time() - start_time

        print('Epoch: {}/{} done. Train: Loss {:.3E} / Error {:.3f} (deg) | Test: Loss {:.3E} / Error {:.3f} (deg). Epoch time: {:.3f} sec.'.format(e+1, args.epochs, train_loss, train_mean_err, test_loss, test_mean_err, elapsed_time))

    if tensorboard_output:
        writer.close()

    return train_stats, test_stats

def main():


    parser = argparse.ArgumentParser(description='Point net experiment')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--batch_size_test', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--enforce_psd', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)


    args = parser.parse_args()
    print(args)

    #Float or Double?
    tensor_type = torch.float


    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float

    if args.cuda:
        pointnet_data = '/home/valentinp/research/RotationContinuity/shapenet/data/pc_plane'
    else:
        pointnet_data = '/Users/valentinp/Dropbox/Postdoc/projects/misc/RotationContinuity/shapenet/data/pc_plane'
    
    train_loader = DataLoader(PointNetDataset(pointnet_data + '/points', rotations_per_batch=100, total_iters=5),
                        batch_size=args.batch_size_train, pin_memory=True, collate_fn=pointnet_collate,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    valid_loader = DataLoader(PointNetDataset('/Users/valentinp/Dropbox/Postdoc/projects/misc/RotationContinuity/shapenet/data/pc_plane/points_test', rotations_per_batch=100, total_iters=5),
                        batch_size=args.batch_size_test, pin_memory=True, collate_fn=pointnet_collate,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    
    #Train and test with new representation
    print('===================TRAINING REP MODEL=======================')
    model_rep = PointNet(dim_out=4, normalize_output=True).to(device=device, dtype=tensor_type)
    loss_fn = quat_squared_loss
    (train_stats_rep, test_stats_rep) = train_test_model(args, loss_fn, model_rep, train_loader, valid_loader)

if __name__=='__main__':
    main()
