import numpy as np
import torch
from liegroups.numpy import SO3
from liegroups.torch import SO3 as SO3_torch
from numpy.linalg import norm
from quaternions import *
from losses import *
from utils import *
from convex_layers import QuadQuatFastSolver, convert_A_to_Avec
from tensorboardX import SummaryWriter
import time
import tqdm

def train_minibatch(model, loss_fn, optimizer, x, targets, A_prior=None):
    #Ensure model gradients are active
    model.train()

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    out = model.forward(x)
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

def train_test_model(args, train_data, test_data, model, loss_fn, rotmat_targets=False, tensorboard_output=True, verbose=False):
    
    if tensorboard_output:
        writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # if pretrain_A_net:
    #     pretrain(A_net, train_data, test_data)

    #Save stats
    train_stats = torch.empty(args.epochs, 2)
    test_stats = torch.empty(args.epochs, 2)

    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float

    pbar = tqdm.tqdm(total=args.epochs)
    for e in range(args.epochs):
        start_time = time.time()

        if args.dataset is not 'static':
            beachball = (args.dataset == 'dynamic_beachball')
            train_data, test_data = create_experimental_data_fast(args.N_train, args.N_test, args.matches_per_sample, sigma=args.sim_sigma, beachball=beachball, device=device, dtype=tensor_type)

        #Train model
        if verbose:
            print('Training...')
        
        num_train_batches = args.N_train // args.batch_size_train
        train_loss = torch.tensor(0.)
        train_mean_err = torch.tensor(0.)
        for k in range(num_train_batches):
            start, end = k * args.batch_size_train, (k + 1) * args.batch_size_train

            if rotmat_targets:
                targets = quat_to_rotmat(train_data.q[start:end])
                (C_est, train_loss_k) = train_minibatch(model, loss_fn, optimizer, train_data.x[start:end], targets)
                train_mean_err += (1/num_train_batches)*rotmat_angle_diff(C_est, targets)
            else:
                targets = train_data.q[start:end]
                (q_est, train_loss_k) = train_minibatch(model, loss_fn, optimizer, train_data.x[start:end], targets)
                train_mean_err += (1/num_train_batches)*quat_angle_diff(q_est, targets)        
        
            train_loss += (1/num_train_batches)*train_loss_k

        #Test model
        if verbose:
            print('Testing...')
        num_test_batches = args.N_test // args.batch_size_test
        test_loss = torch.tensor(0.)
        test_mean_err = torch.tensor(0.)


        for k in range(num_test_batches):
            start, end = k * args.batch_size_test, (k + 1) * args.batch_size_test

            if rotmat_targets:
                targets = quat_to_rotmat(test_data.q[start:end])
                (C_est, test_loss_k) =  test_model(model, loss_fn, test_data.x[start:end], targets)
                test_mean_err += (1/num_test_batches)*rotmat_angle_diff(C_est, targets)
            else:
                targets = test_data.q[start:end]
                (q_est, test_loss_k) =  test_model(model, loss_fn, test_data.x[start:end], targets)
                test_mean_err += (1/num_test_batches)*quat_angle_diff(q_est, targets)   

            test_loss += (1/num_test_batches)*test_loss_k

        #scheduler.step()

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
        
        if verbose:
            print('Epoch: {}/{}. Train: Loss {:.3E} / Error {:.3f} (deg) | Test: Loss {:.3E} / Error {:.3f} (deg). Epoch time: {:.3f} sec.'.format(e+1, args.epochs, train_loss, train_mean_err, test_loss, test_mean_err, elapsed_time))
        
        output_string = 'Epoch: {}/{}. Train: Loss {:.3E} / Error {:.3f} (deg) | Test: Loss {:.3E} / Error {:.3f} (deg). Epoch time: {:.3f} sec.'.format(e+1, args.epochs, train_loss, train_mean_err, test_loss, test_mean_err, elapsed_time)
        pbar.set_description(output_string)
        pbar.update(1)
    
    pbar.close()
    if tensorboard_output:
        writer.close()

    return train_stats, test_stats

    
def build_A(x_1, x_2, sigma_2):
    N = x_1.shape[0]
    A = np.zeros((4, 4), dtype=np.float64)
    for i in range(N):
        # Block diagonal indices
        I = np.eye(4, dtype=np.float64)
        t1 = (x_2[i].dot(x_2[i]) + x_1[i].dot(x_1[i]))*I
        t2 = 2.*Omega_l(pure_quat(x_2[i])).dot(
            Omega_r(pure_quat(x_1[i])))
        A_i = (t1 + t2)/(sigma_2[i])
        A += A_i
    return A 

#Note sigma can be scalar or an N-dimensional vector of std. devs.
def gen_sim_data(N, sigma, torch_vars=False, shuffle_points=False):
    ##Simulation
    #Create a random rotation
    C = SO3.exp(np.random.randn(3)).as_matrix()
    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = normalized(np.random.randn(N, 3), axis=1)
    #Rotate and add noise
    noise = np.random.randn(N,3)
    noise = (noise.T*sigma).T
    x_2 = C.dot(x_1.T).T + noise

    if shuffle_points:
        x_1, x_2 = unison_shuffled_copies(x_1,x_2)

    if torch_vars:
        C = torch.from_numpy(C)
        x_1 = torch.from_numpy(x_1)
        x_2 = torch.from_numpy(x_2)

    return C, x_1, x_2

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def gen_sim_data_grid(N, sigma, torch_vars=False, shuffle_points=False):
    ##Simulation
    #Create a random rotation
    C = SO3.exp(np.random.randn(3)).as_matrix()
    
    #Grid is fixed 
    grid_dim = 50
    xlims = np.linspace(-1., 1., grid_dim)
    ylims = np.linspace(-1., 1., grid_dim)
    x, y = np.meshgrid(xlims, ylims)
    z = np.sin(x)*np.cos(y)
    x_1 =  normalized(np.hstack((x.reshape(grid_dim**2, 1), y.reshape(grid_dim**2, 1), z.reshape(grid_dim**2, 1))), axis=1)
    
    #Sample N points
    ids = np.random.permutation(x_1.shape[0])
    x_1 = x_1[ids[:N]]

    #Sort into canonical order
    #x_1 = x_1[x_1[:,0].argsort()]

    #Rotate and add noise
    noise = np.random.randn(N,3)
    noise = (noise.T*sigma).T
    x_2 = C.dot(x_1.T).T + noise

    if shuffle_points:
        x_1, x_2 = unison_shuffled_copies(x_1,x_2)


    if torch_vars:
        C = torch.from_numpy(C)
        x_1 = torch.from_numpy(x_1)
        x_2 = torch.from_numpy(x_2)

    return C, x_1, x_2

class SyntheticData():
    def __init__(self, x, q, A_prior):
        self.x = x
        self.q = q
        self.A_prior = A_prior


def gen_sim_data_fast(N_rotations, N_matches_per_rotation, sigma, dtype=torch.double):
    ##Simulation
    #Create a random rotation
    C = SO3_torch.exp(torch.randn(N_rotations, 3, dtype=dtype)).as_matrix()
    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = torch.randn(N_rotations, 3, N_matches_per_rotation, dtype=dtype)
    x_1 = x_1/x_1.norm(dim=1,keepdim=True)    
    #Rotate and add noise
    noise = sigma*torch.randn_like(x_1)
    x_2 = C.bmm(x_1) + noise
    return C, x_1, x_2

def gen_sim_data_beachball(N_rotations, N_matches_per_rotation, sigma, dtype=torch.double):
    ##Simulation
    #Create a random rotation
    C = SO3_torch.exp(torch.randn(N_rotations, 3, dtype=dtype)).as_matrix()
    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = torch.randn(3, N_rotations*N_matches_per_rotation, dtype=dtype)
    x_1 = x_1/x_1.norm(dim=0,keepdim=True)

    #x_1 = torch.randn(N_rotations, 3, N_matches_per_rotation, dtype=dtype)

    region_masks = [(x_1[0] < 0.) & (x_1[1] < 0.), 
                (x_1[0] >= 0.) & (x_1[1] < 0.), 
                (x_1[0] < 0.) & (x_1[1] >= 0.), 
                (x_1[0] >= 0.) & (x_1[1] >= 0.)]
    sigma_list = [0.1*sigma, 0.5*sigma, 2*sigma, 10*sigma]

    noise = torch.zeros_like(x_1)
    for r_i, region in enumerate(region_masks):
        noise[:, region] = sigma_list[r_i]*torch.randn_like(noise[:, region])

    x_1 = x_1.view(3, N_rotations, N_matches_per_rotation).transpose(0,1) 
    noise = noise.view(3, N_rotations, N_matches_per_rotation).transpose(0,1) 

    #Rotate and add noise
    x_2 = C.bmm(x_1) + noise
    return C, x_1, x_2

def create_experimental_data_fast(N_train=2000, N_test=50, N_matches_per_sample=100, sigma=0.01, beachball=False, device=torch.device('cpu'), dtype=torch.double):
    
    if beachball:
        C_train, x_1_train, x_2_train = gen_sim_data_beachball(N_train, N_matches_per_sample, sigma)
        C_test, x_1_test, x_2_test = gen_sim_data_beachball(N_test, N_matches_per_sample, sigma)
    else:
        C_train, x_1_train, x_2_train = gen_sim_data_fast(N_train, N_matches_per_sample, sigma)
        C_test, x_1_test, x_2_test = gen_sim_data_fast(N_test, N_matches_per_sample, sigma)

    x_train = torch.empty(N_train, 2, N_matches_per_sample, 3, dtype=dtype, device=device)
    x_train[:,0,:,:] = x_1_train.transpose(1,2)
    x_train[:,1,:,:] = x_2_train.transpose(1,2)
    
    q_train = rotmat_to_quat(C_train, ordering='xyzw').to(dtype=dtype, device=device)

    x_test = torch.empty(N_test, 2, N_matches_per_sample, 3, dtype=dtype, device=device)
    x_test[:,0,:,:] = x_1_test.transpose(1,2)
    x_test[:,1,:,:] = x_2_test.transpose(1,2)
    
    q_test = rotmat_to_quat(C_test, ordering='xyzw').to(dtype=dtype, device=device)

    train_data = SyntheticData(x_train, q_train, None)
    test_data = SyntheticData(x_test, q_test, None)
    
    return train_data, test_data    


def create_experimental_data(N_train=2000, N_test=50, N_matches_per_sample=100, sigma=0.01, device=torch.device('cpu'), dtype=torch.double):

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

        C, x_1, x_2 = gen_sim_data(N_matches_per_sample, sigma_sim_vec, torch_vars=True, shuffle_points=False)
        q = rotmat_to_quat(C, ordering='xyzw')
        x_train[n, 0, :, :] = x_1
        x_train[n, 1, :, :] = x_2
        q_train[n] = q
        A_prior_train[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))

    for n in range(N_test):
        C, x_1, x_2 = gen_sim_data(N_matches_per_sample, sigma_sim_vec, torch_vars=True, shuffle_points=False)
        q = rotmat_to_quat(C, ordering='xyzw')
        x_test[n, 0, :, :] = x_1
        x_test[n, 1, :, :] = x_2
        q_test[n] = q
        A_prior_test[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))

        # A_vec = convert_A_to_Avec(A_prior_test[n]).unsqueeze(dim=0)
        # print(q - QuadQuatFastSolver.apply(A_vec).squeeze())
    

    x_train = x_train.to(device=device)
    q_train = q_train.to(device=device)
    A_prior_train = A_prior_train.to(device=device)
    x_test = x_test.to(device=device)
    q_test = q_test.to(device=device)
    A_prior_test = A_prior_test.to(device=device)

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
        q_est = rotmat_to_quat(C, ordering='xyzw')
        err[i] = quat_angle_diff(q_est, sim_data.q[i])
    return err.mean()
