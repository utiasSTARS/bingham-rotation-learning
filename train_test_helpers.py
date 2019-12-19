import torch
import time, argparse
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter
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


def train_test_model(args, loss_fn, model, train_loader, test_loader, tensorboard_output=True, progress_bar=True):

    if tensorboard_output:
        writer = SummaryWriter()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    #Save stats
    train_stats = torch.zeros(args.epochs, 2)
    test_stats = torch.zeros(args.epochs, 2)
    
    device = next(model.parameters()).device

    rotmat_targets = train_loader.dataset.rotmat_targets

    for e in range(args.epochs):
        start_time = time.time()

        #Train model
        model.train()
        train_loss = torch.tensor(0.)
        train_mean_err = torch.tensor(0.)
        num_train_batches = len(train_loader)

        if progress_bar:
            pbar = tqdm.tqdm(total=num_train_batches)

        for _, (x, target) in enumerate(train_loader):
            #Move all data to appropriate device
            target = target.to(device)
            x = x.to(device)
            (rot_est, train_loss_k) = train(model, loss_fn, optimizer, x, target)
            if rotmat_targets:
                q_est = rotmat_to_quat(rot_est)
                q_gt = rotmat_to_quat(target)
            else:
                q_est = rot_est
                q_gt = target
            train_loss += (1./num_train_batches)*train_loss_k
            train_mean_err += (1./num_train_batches)*quat_angle_diff(q_est, q_gt)
            if progress_bar:
                pbar.update(1)
        
        if progress_bar:
            pbar.close()

        #Test model
        model.eval()
        num_test_batches = len(test_loader)
        test_loss = torch.tensor(0.)
        test_mean_err = torch.tensor(0.)

        for _, (x, target) in enumerate(test_loader):
            #Move all data to appropriate device
            target = target.to(device)
            x = x.to(device)
            (rot_est, test_loss_k) = test(model, loss_fn, x, target)

            if rotmat_targets:
                test_mean_err += (1./num_test_batches)*rotmat_angle_diff(rot_est, target)
            else:
                test_mean_err += (1./num_test_batches)*quat_angle_diff(rot_est, target)

            test_loss += (1./num_test_batches)*test_loss_k

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
        print(output_string)

    if tensorboard_output:
        writer.close()
    return train_stats, test_stats