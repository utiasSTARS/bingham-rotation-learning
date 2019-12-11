import torch
import time
import numpy as np
from networks import *
from quaternions import *
from sim_helpers import *
from convex_layers import QuadQuatFastSolver, convert_A_to_Avec
from tensorboardX import SummaryWriter
from datetime import datetime
import argparse

#Generic training function
def train_minibatch(model, loss_fn, optimizer, x, targets, A_prior=None):
    #Ensure model gradients are active
    model.train()

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    out = model.forward(x, A_prior)
    loss = loss_fn(out, targets)

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    return (out, loss.item())

def test_model(model, loss_fn, x, targets, **kwargs):
    #model.eval() speeds things up because it turns off gradient computation
    model.eval()
    # Forward
    with torch.no_grad():
        out = model.forward(x, **kwargs)
        loss = loss_fn(out, targets)

    return (out, loss.item())

def pretrain(A_net, train_data, test_data):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(A_net.parameters(), lr=1e-2)
    batch_size = 50
    num_epochs = 500

    print('Pre-training A network...')
    N_train = train_data.x.shape[0]
    N_test = test_data.x.shape[0]
    num_train_batches = N_train // batch_size
    for e in range(num_epochs):
        start_time = time.time()

        #Train model
        train_loss = torch.tensor(0.)
        for k in range(num_train_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            _, train_loss_k = train_minibatch(A_net, loss_fn, optimizer,  train_data.x[start:end], convert_A_to_Avec(train_data.A_prior[start:end]))
            train_loss += (1/num_train_batches)*train_loss_k
    
        elapsed_time = time.time() - start_time

        #Test model
        num_test_batches = N_test // batch_size
        test_loss = torch.tensor(0.)
        for k in range(num_test_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            _, test_loss_k = test_model(A_net, loss_fn, test_data.x[start:end], convert_A_to_Avec(test_data.A_prior[start:end]))
            test_loss += (1/num_test_batches)*test_loss_k


        print('Epoch: {}/{}. Train: Loss {:.3E} | Test: Loss {:.3E}. Epoch time: {:.3f} sec.'.format(e+1, num_epochs, train_loss, test_loss, elapsed_time))

    return

def train_test_model(args, train_data, test_data, model, tensorboard_output=True):

    if args.bidirectional_loss:
        loss_fn = quat_consistency_loss
    else:
        loss_fn = quat_squared_loss
    
    if tensorboard_output:
        writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    # if pretrain_A_net:
    #     pretrain(A_net, train_data, test_data)

    #Save stats
    train_stats = torch.empty(args.total_epochs, 2)
    test_stats = torch.empty(args.total_epochs, 2)
    
    for e in range(args.total_epochs):
        start_time = time.time()


        #Train model
        print('Training...')
        num_train_batches = args.N_train // args.batch_size_train
        train_loss = torch.tensor(0.)
        train_mean_err = torch.tensor(0.)
        for k in range(num_train_batches):
            start, end = k * args.batch_size_train, (k + 1) * args.batch_size_train

            if args.use_A_prior:
                A_prior = convert_A_to_Avec(train_data.A_prior[start:end])
            else:
                A_prior = None
            
            (q_est, train_loss_k) = train_minibatch(model, loss_fn, optimizer, train_data.x[start:end], train_data.q[start:end], A_prior=A_prior)
            q_train = q_est[0] if args.bidirectional_loss else q_est
            train_loss += (1/num_train_batches)*train_loss_k
            train_mean_err += (1/num_train_batches)*quat_angle_diff(q_train, train_data.q[start:end])
        

        #Test model
        print('Testing...')
        num_test_batches = args.N_test // args.batch_size_test
        test_loss = torch.tensor(0.)
        test_mean_err = torch.tensor(0.)


        for k in range(num_test_batches):
            start, end = k * args.batch_size_test, (k + 1) * args.batch_size_test
            if args.use_A_prior:
                A_prior = convert_A_to_Avec(test_data.A_prior[start:end])
            else:
                A_prior = None
            (q_est, test_loss_k) = test_model(model, loss_fn, test_data.x[start:end], test_data.q[start:end], A_prior=A_prior)
            q_test = q_est[0] if args.bidirectional_loss else q_est
            test_loss += (1/num_test_batches)*test_loss_k
            test_mean_err += (1/num_test_batches)*quat_angle_diff(q_test, test_data.q[start:end])


        #scheduler.step(test_loss)

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


    parser = argparse.ArgumentParser(description='Synthetic Wahba arguments.')
    parser.add_argument('--sim_sigma', type=float, default=1e-6)
    parser.add_argument('--N_train', type=int, default=5000)
    parser.add_argument('--N_test', type=int, default=100)
    parser.add_argument('--matches_per_sample', type=int, default=100)

    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--batch_size_train', type=int, default=500)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--bidirectional_loss', action='store_true', default=False)
    parser.add_argument('--pretrain_A_net', action='store_true', default=False)
    parser.add_argument('--use_A_prior', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)


    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float


    #Generate data
    train_data, test_data = create_experimental_data(args.N_train, args.N_test, args.matches_per_sample, sigma=args.sim_sigma, device=device, dtype=tensor_type)
    
    

    #Train and test direct model
    print('===================TRAINING DIRECT MODEL=======================')
    model_direct = QuatNetDirect(num_pts=args.matches_per_sample).to(device=device, dtype=tensor_type)
    (train_stats_direct, test_stats_direct) = train_test_model(args, train_data, test_data, model_direct, tensorboard_output=True)

    #Train and test with new representation
    print('===================TRAINING REP MODEL=======================')
    A_net = ANet(num_pts=args.matches_per_sample, bidirectional=args.bidirectional_loss).to(device=device, dtype=tensor_type)
    model_rep = QuatNet(A_net=A_net).to(device=device, dtype=tensor_type)
    (train_stats_rep, test_stats_rep) = train_test_model(args, train_data, test_data, model_rep, tensorboard_output=True)

    
    saved_data_file_name = 'synthetic_wahba_experiment_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    full_saved_path = 'saved_data/synthetic/{}.pt'.format(saved_data_file_name)
    torch.save({
            'model_rep': model_rep.state_dict().detach().cpu(),
            'model_direct': model_direct.state_dict().detach().cpu(),
            'train_stats_direct': train_stats_direct.detach().cpu(),
            'test_stats_direct': test_stats_direct.detach().cpu(),
            'train_stats_rep': train_stats_rep.detach().cpu(),
            'test_stats_rep': test_stats_rep.detach().cpu(),
            'args': args,
        }, full_saved_path)

    print('Saved data to {}.'.format(full_saved_path))

if __name__=='__main__':
    main()
