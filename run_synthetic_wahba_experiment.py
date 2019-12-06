import torch
import time
import numpy as np
from sim_models import *
from quaternions import *
from sim_helpers import *
from convex_layers import QuadQuatFastSolver
from tensorboardX import SummaryWriter
from datetime import datetime
import argparse
from liegroups.torch import SO3


class SyntheticData():
    def __init__(self, x, q, A_prior):
        self.x = x
        self.q = q
        self.A_prior = A_prior


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



def create_experimental_data(N_train=2000, N_test=50, N_matches_per_sample=100, sigma=0.01, dtype=torch.double):

    x_train = torch.empty(N_train, 2, N_matches_per_sample, 3, dtype=dtype)
    q_train = torch.empty(N_train, 4, dtype=dtype)
    A_prior_train = torch.empty(N_train, 4, 4, dtype=dtype)

    x_test = torch.empty(N_test, 2, N_matches_per_sample, 3, dtype=dtype)
    q_test = torch.empty(N_test, 4, dtype=dtype)
    A_prior_test = torch.empty(N_test, 4, 4, dtype=dtype)

    sigma_sim_vec = sigma*np.ones(N_matches_per_sample)
    #sigma_sim_vec[:int(N_matches_per_sample/2)] *= 10 #Artificially scale half the noise
    sigma_prior_vec = sigma*np.ones(N_matches_per_sample)
    

    for n in range(N_train):

        C, x_1, x_2 = gen_sim_data_grid(N_matches_per_sample, sigma_sim_vec, torch_vars=True, shuffle_points=False)
        q = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        x_train[n, 0, :, :] = x_1
        x_train[n, 1, :, :] = x_2
        q_train[n] = q
        A_prior_train[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))

    for n in range(N_test):
        C, x_1, x_2 = gen_sim_data_grid(N_matches_per_sample, sigma_sim_vec, torch_vars=True, shuffle_points=False)
        q = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        x_test[n, 0, :, :] = x_1
        x_test[n, 1, :, :] = x_2
        q_test[n] = q
        A_prior_test[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))

        # A_vec = convert_A_to_Avec(A_prior_test[n]).unsqueeze(dim=0)
        # print(q - QuadQuatFastSolver.apply(A_vec).squeeze())
    
    train_data = SyntheticData(x_train, q_train, A_prior_train)
    test_data = SyntheticData(x_test, q_test, A_prior_test)
    
    return train_data, test_data


def compute_mean_horn_error(sim_data):
    N = sim_data.x.shape[0]
    err = torch.empty(N)
    for i in range(N):
        x = sim_data.x[i]
        x_1 = x[0,:,:].numpy()
        x_2 = x[1,:,:].numpy()
        C = torch.from_numpy(solve_horn(x_1, x_2))
        q_est = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        err[i] = quat_angle_diff(q_est, sim_data.q[i])
    return err.mean()

def convert_A_to_Avec(A):
    if A.dim() < 3:
        A = A.unsqueeze(dim=0)
    idx = torch.triu_indices(4,4)
    A_vec = A[:, idx[0], idx[1]]

    A_vec = A_vec/A_vec.norm(dim=1).view(-1, 1)


    return A_vec.squeeze()


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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)

    # if pretrain_A_net:
    #     pretrain(A_net, train_data, test_data)

    #Save stats
    train_stats = torch.empty(args.total_epochs, 2)
    test_stats = torch.empty(args.total_epochs, 2)
    
    for e in range(args.total_epochs):
        start_time = time.time()


        #Train model
        print('Training... lr: {:.3E}'.format(scheduler.get_lr()[0]))
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
        
        scheduler.step()

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
    parser.add_argument('--sim_sigma', type=float, default=1e-2)
    parser.add_argument('--N_train', type=int, default=1000)
    parser.add_argument('--N_test', type=int, default=500)
    parser.add_argument('--matches_per_sample', type=int, default=100)

    parser.add_argument('--total_epochs', type=int, default=10)
    parser.add_argument('--batch_size_train', type=int, default=250)
    parser.add_argument('--batch_size_test', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--bidirectional_loss', action='store_true', default=False)
    parser.add_argument('--pretrain_A_net', action='store_true', default=False)
    parser.add_argument('--use_A_prior', action='store_true', default=False)


    args = parser.parse_args()
    print(args)

    #Generate data
    train_data, test_data = create_experimental_data(args.N_train, args.N_test, args.matches_per_sample, sigma=args.sim_sigma)
    
    #Train and test direct model
    print('===================TRAINING DIRECT MODEL=======================')
    model_direct = QuatNetDirect(num_pts=args.matches_per_sample).double()
    (train_stats_direct, test_stats_direct) = train_test_model(args, train_data, test_data, model_direct, tensorboard_output=True)

    #Train and test with new representation
    print('===================TRAINING REP MODEL=======================')
    A_net = ANet(num_pts=args.matches_per_sample, bidirectional=args.bidirectional_loss).double()
    model_rep = QuatNet(A_net=A_net)
    (train_stats_rep, test_stats_rep) = train_test_model(args, train_data, test_data, model_rep, tensorboard_output=True)

    
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
