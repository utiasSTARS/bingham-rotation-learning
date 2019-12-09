import torch
import time, datetime, argparse
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime
from loaders import SevenScenesData
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from quaternions import *
from networks import *

#Generic training function
def train(model, loss_fn, optimizer, im, q_gt):

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    q_est = model.forward(im)
    loss = loss_fn(q_est, q_gt)

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    return (q_est, loss.item())


def test(model, loss_fn, im, q_gt):
    # Forward
    with torch.no_grad():
        q_est = model.forward(im)
        loss = loss_fn(q_est, q_gt)
            
    return (q_est, loss.item())


def train_test_model(args, loss_fn, model, train_loader, test_loader, tensorboard_output=True):

    if tensorboard_output:
        writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)

    #Save stats
    train_stats = torch.empty(args.total_epochs, 2)
    test_stats = torch.empty(args.total_epochs, 2)
    
    device = next(model.parameters()).device


    for e in range(args.total_epochs):
        start_time = time.time()


        #Train model
        print('Training... lr: {:.3E}'.format(scheduler.get_lr()[0]))
        model.train()
        num_train_batches = len(train_loader)
        train_loss = torch.tensor(0.)
        train_mean_err = torch.tensor(0.)

        for batch_idx, (im, q_gt) in enumerate(train_loader):
            #Move all data to appropriate device
            q_gt.to(device)
            if isinstance(im, list):
                im[0] = im[0].to(device)
                im[1] = im[1].to(device)
            else:
                im = im.to(device)

            (q_est, train_loss_k) = train(model, loss_fn, optimizer, im, q_gt)
            train_loss += (1./num_train_batches)*train_loss_k
            train_mean_err += (1./num_train_batches)*quat_angle_diff(q_est, q_gt)
            print('batch: {}/{}'.format(batch_idx, num_train_batches))
            
        scheduler.step()

        #Test model
        print('Testing...')
        model.eval()
        num_test_batches = len(test_loader)
        test_loss = torch.tensor(0.)
        test_mean_err = torch.tensor(0.)


        for batch_idx, (im, q_gt) in enumerate(test_loader):
            #Move all data to appropriate device
            q_gt.to(device)
            if isinstance(im, list):
                im[0] = im[0].to(device)
                im[1] = im[1].to(device)
            else:
                im = im.to(device)

            (q_est, test_loss_k) = test(model, loss_fn, im, q_gt)
            test_loss += (1./num_test_batches)*test_loss_k
            test_mean_err += (1./num_test_batches)*quat_angle_diff(q_est, q_gt)
            print('batch: {}/{}'.format(batch_idx, num_train_batches))

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

        print('Epoch: {}/{}. Train: Loss {:.3E} / Error {:.3f} (deg) | Test: Loss {:.3E} / Error {:.3f} (deg). Epoch time: {:.3f} sec.'.format(e+1, args.total_epochs, train_loss, train_mean_err, test_loss, test_mean_err, elapsed_time))

    if tensorboard_output:
        writer.close()

    return train_stats, test_stats

def main():


    parser = argparse.ArgumentParser(description='7Scenes experiment')
    parser.add_argument('--scene', type=str, default='chess')
    parser.add_argument('--total_epochs', type=int, default=20)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_test', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    print(args)

    #Float or Double?
    tensor_type = torch.float


    #Load datasets
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not args.cuda:
        data_folder = '/Users/valentinp/Desktop/datasets/7scenes'
        device = torch.device('cpu')
    else:
        data_folder = '/media/m2-drive/datasets/7scenes'
        device = torch.device('cuda:0')

    train_loader = DataLoader(SevenScenesData(args.scene, data_folder, train=True, transform=transform),
                        batch_size=args.batch_size_train, pin_memory=True,
                        shuffle=True, num_workers=4, drop_last=False)
    valid_loader = DataLoader(SevenScenesData(args.scene, data_folder, train=False, transform=transform),
                        batch_size=args.batch_size_test, pin_memory=True,
                        shuffle=False, num_workers=4, drop_last=False)
    
    #Train and test direct model
    # print('===================TRAINING DIRECT MODEL=======================')
    model_direct = CustomResNetDirect()
    model_direct.to(dtype=tensor_type, device=device)
    loss_fn = quat_squared_loss
    (train_stats_direct, test_stats_direct) = train_test_model(args, loss_fn, model_direct, train_loader, train_loader)

    #Train and test with new representation
    print('===================TRAINING REP MODEL=======================')
    model_rep = CustomResNetConvex()
    model_rep.to(dtype=tensor_type, device=device)
    loss_fn = quat_squared_loss
    (train_stats_rep, test_stats_rep) = train_test_model(args, loss_fn, model_rep, train_loader, valid_loader)

    
    saved_data_file_name = '7scenes_experiment_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = 'saved_data/7scenes/{}.pt'.format(saved_data_file_name)
    torch.save({
            # 'model_rep': model_rep.state_dict(),
            # 'model_direct': model_direct.state_dict(),
            'train_stats_direct': train_stats_direct.detach().cpu(),
            'test_stats_direct': test_stats_direct.detach().cpu(),
            'train_stats_rep': train_stats_rep.detach().cpu(),
            'test_stats_rep': test_stats_rep.detach().cpu(),
            'args': args,
        }, full_saved_path)

    print('Saved data to {}.'.format(full_saved_path))

if __name__=='__main__':
    main()
