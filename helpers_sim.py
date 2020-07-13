import numpy as np
import torch
from liegroups.numpy import SO3
from liegroups.torch import SO3 as SO3_torch
from numpy.linalg import norm
from quaternions import *
from losses import *
from utils import *
from qcqp_layers import QuadQuatFastSolver, convert_A_to_Avec
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
            beachball_factors = args.beachball_sigma_factors
            train_data, test_data = create_experimental_data_fast(args.N_train, args.N_test, args.matches_per_sample, max_rotation_angle=args.max_rotation_angle, sigma=args.sim_sigma, beachball=beachball, beachball_factors=beachball_factors, device=device, dtype=tensor_type)

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


def train_test_models_with_plots(args, train_data, test_data, models, loss_fns, rotmat_targets, verbose=False):
    """
    Helper for rss_demo.ipynb
    :param args:
    :param train_data:
    :param test_data:
    :param models:
    :param loss_fn:
    :param rotmat_targets:
    :param verbose:
    :return:
    """
    # from jupyterplot import ProgressPlot
    from lrcurve.plot_learning_curve import PlotLearningCurve
    # from matplotlib import pyplot as plt
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in models]

    # Save stats for plotting
    train_stats = torch.empty(len(models), args.epochs, 2)
    test_stats = torch.empty(len(models), args.epochs, 2)

    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float

    # JupyterPlot way (broken!)
    # pp_train = ProgressPlot(line_names=["Quaternion", "6D", "Bingham"], x_lim=[0, args.epochs])
    # pp_test = ProgressPlot(line_names=["Quaternion", "6D", "Bingham"], x_lim=[0, args.epochs])
    # pp = ProgressPlot(plot_names=["Train", "Test"], line_names=["Quaternion", "6D", "Bingham"],
    #                   x_lim=[0, args.epochs], y_lim=[-3, 3])
    # lrcurve way
    # plot = PlotLearningCurve()
    plot = PlotLearningCurve(
        line_config={
            'train_quat': {'name': 'quat', 'color': '#F90909', 'facet': 'train'},
            'train_6d': {'name': '6D', 'color': '#63FF14', 'facet': 'train'},
            'train_bing': {'name': 'bing', 'color': '#00BFC4', 'facet': 'train'},
            'test_quat': {'name': 'quat', 'color': '#F90909', 'facet': 'test'},
            'test_6d': {'name': '6D', 'color': '#63FF14', 'facet': 'test'},
            'test_bing': {'name': 'bing', 'color': '#00BFC4', 'facet': 'test'}
        },
        facet_config={
            'train': {'name': 'Training', 'limit': [-3, 3], 'scale': 'log'},
            'test': {'name': 'Testing', 'limit': [-3, 3], 'scale': 'log'}
        },
        xaxis_config={'name': 'Epoch', 'limit': [0, args.epochs]}
    )
    # PyPlot way
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.ion()

    # fig.show()
    # fig.canvas.draw()

    for e in range(args.epochs):
        start_time = time.time()

        if args.dataset is not 'static':
            beachball = (args.dataset == 'dynamic_beachball')
            beachball_factors = args.beachball_sigma_factors
            train_data, test_data = create_experimental_data_fast(args.N_train, args.N_test, args.matches_per_sample,
                                                                  max_rotation_angle=args.max_rotation_angle,
                                                                  sigma=args.sim_sigma, beachball=beachball,
                                                                  beachball_factors=beachball_factors, device=device,
                                                                  dtype=tensor_type)

        num_train_batches = args.N_train // args.batch_size_train
        train_loss = torch.zeros(len(models))
        train_mean_err = torch.zeros(len(models))
        for idx, (model, optimizer, loss_fn, rotmat_target) in enumerate(zip(models, optimizers, loss_fns, rotmat_targets)):
            for k in range(num_train_batches):
                start, end = k * args.batch_size_train, (k + 1) * args.batch_size_train
    
                if rotmat_target:
                    targets = quat_to_rotmat(train_data.q[start:end])
                    (C_est, train_loss_k) = train_minibatch(model, loss_fn, optimizer, train_data.x[start:end], targets)
                    train_mean_err += (1 / num_train_batches) * rotmat_angle_diff(C_est, targets)
                else:
                    targets = train_data.q[start:end]
                    (q_est, train_loss_k) = train_minibatch(model, loss_fn, optimizer, train_data.x[start:end], targets)
                    train_mean_err[idx] += (1 / num_train_batches) * quat_angle_diff(q_est, targets)
    
                train_loss[idx] += (1 / num_train_batches) * train_loss_k

        # Test model
        if verbose:
            print('Testing...')
        num_test_batches = args.N_test // args.batch_size_test
        test_loss = torch.zeros(len(models))
        test_mean_err = torch.zeros(len(models))
        for idx, (model, loss_fn, rotmat_target) in enumerate(zip(models, loss_fns, rotmat_targets)):
            for k in range(num_test_batches):
                start, end = k * args.batch_size_test, (k + 1) * args.batch_size_test
    
                if rotmat_target:
                    targets = quat_to_rotmat(test_data.q[start:end])
                    (C_est, test_loss_k) = test_model(model, loss_fn, test_data.x[start:end], targets)
                    test_mean_err += (1 / num_test_batches) * rotmat_angle_diff(C_est, targets)
                else:
                    targets = test_data.q[start:end]
                    (q_est, test_loss_k) = test_model(model, loss_fn, test_data.x[start:end], targets)
                    test_mean_err[idx] += (1 / num_test_batches) * quat_angle_diff(q_est, targets)
    
                test_loss[idx] += (1 / num_test_batches) * test_loss_k

        # History tracking
        train_stats[:, e, 0] = train_loss
        train_stats[:, e, 1] = train_mean_err
        test_stats[:, e, 0] = test_loss
        test_stats[:, e, 1] = test_mean_err

        # pp.update([list(np.log10(train_loss)),
        #            list(np.log10(test_loss))])

        # plot.append(e, {
        #     'train': {
        #         'Quat.': train_loss[0],
        #         '6D': train_loss[1],
        #         'Bing.': train_loss[2]
        #     }
        # })
        # plot.append(e, {
        #     'test': {
        #         'Quat.': test_loss[0],
        #         '6D': test_loss[1],
        #         'Bing.': test_loss[2]
        #     }
        # })
        plot.append(e, {
            'train': {'train_quat': train_loss[0]},
            'train': {'train_6d': train_loss[1]},
            'train': {'train_bing': train_loss[2]},
            'test': {'test_quat': test_loss[0]},
            'test': {'test_6d': test_loss[1]},
            'test': {'test_bing': test_loss[2]}
        })
        plot.draw()

        # ax.plot(np.arange(e, e+1), train_loss[0, 0:e, 0], 'r-')
        # fig.canvas.draw()

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


def gen_sim_data_fast(N_rotations, N_matches_per_rotation, sigma, max_rotation_angle=None, dtype=torch.double):
    ##Simulation
    #Create a random rotation
    axis = torch.randn(N_rotations, 3, dtype=dtype)
    axis = axis / axis.norm(dim=1, keepdim=True)
    if max_rotation_angle:
        max_angle = max_rotation_angle*np.pi/180.
    else:
        max_angle = np.pi
    
    angle = max_angle*torch.rand(N_rotations, 1)

    C = SO3_torch.exp(angle*axis).as_matrix()
    if N_rotations == 1:
        C = C.unsqueeze(dim=0)
    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = torch.randn(N_rotations, 3, N_matches_per_rotation, dtype=dtype)
    x_1 = x_1/x_1.norm(dim=1,keepdim=True)   
    #Rotate and add noise
    noise = sigma*torch.randn_like(x_1)
    x_2 = C.bmm(x_1) + noise
    
    return C, x_1, x_2

def gen_sim_data_beachball(N_rotations, N_matches_per_rotation, sigma, factors, dtype=torch.double):
    ##Simulation
    #Create a random rotation
    C = SO3_torch.exp(torch.randn(N_rotations, 3, dtype=dtype)).as_matrix()
    #Create two sets of vectors (normalized to unit l2 norm)
    x_1 = torch.randn(3, N_rotations*N_matches_per_rotation, dtype=dtype)
    x_1 = x_1/x_1.norm(dim=0,keepdim=True)

    region_masks = [(x_1[0] < 0.) & (x_1[1] < 0.), 
                (x_1[0] >= 0.) & (x_1[1] < 0.), 
                (x_1[0] < 0.) & (x_1[1] >= 0.), 
                (x_1[0] >= 0.) & (x_1[1] >= 0.)]

    noise = torch.zeros_like(x_1)
    for r_i, region in enumerate(region_masks):
        noise[:, region] = factors[r_i]*sigma*torch.randn_like(noise[:, region])

    x_1 = x_1.view(3, N_rotations, N_matches_per_rotation).transpose(0,1) 
    noise = noise.view(3, N_rotations, N_matches_per_rotation).transpose(0,1) 

    
    #Rotate and add noise
    x_2 = C.bmm(x_1) + noise
    return C, x_1, x_2

def create_experimental_data_fast(N_train=2000, N_test=50, N_matches_per_sample=100, sigma=0.01, beachball=False, max_rotation_angle=None, beachball_factors=None, device=torch.device('cpu'), dtype=torch.double):
    
    if beachball:
        C_train, x_1_train, x_2_train = gen_sim_data_beachball(N_train, N_matches_per_sample, sigma, beachball_factors)
        C_test, x_1_test, x_2_test = gen_sim_data_beachball(N_test, N_matches_per_sample, sigma, beachball_factors)
    else:
        C_train, x_1_train, x_2_train = gen_sim_data_fast(N_train, N_matches_per_sample, sigma, max_rotation_angle=max_rotation_angle)
        C_test, x_1_test, x_2_test = gen_sim_data_fast(N_test, N_matches_per_sample, sigma, max_rotation_angle=max_rotation_angle)

    x_train = torch.empty(N_train, 2, N_matches_per_sample, 3, dtype=dtype, device=device)
    x_train[:,0,:,:] = x_1_train.transpose(1,2)
    x_train[:,1,:,:] = x_2_train.transpose(1,2)
    
    q_train = rotmat_to_quat(C_train, ordering='xyzw').to(dtype=dtype, device=device)
    if q_train.dim() < 2:
        q_train = q_train.unsqueeze(dim=0)


    x_test = torch.empty(N_test, 2, N_matches_per_sample, 3, dtype=dtype, device=device)
    x_test[:,0,:,:] = x_1_test.transpose(1,2)
    x_test[:,1,:,:] = x_2_test.transpose(1,2)
    
    q_test = rotmat_to_quat(C_test, ordering='xyzw').to(dtype=dtype, device=device)
    if q_test.dim() < 2:
        q_test = q_test.unsqueeze(dim=0)
    
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


if __name__ == '__main__':

    import torch
    import numpy as np
    from networks import *
    from losses import *
    # from helpers_sim import *
    import argparse

    parser = argparse.ArgumentParser(description='Synthetic Wahba arguments.')
    parser.add_argument('--sim_sigma', type=float, default=1e-2)
    parser.add_argument('--N_train', type=int, default=500)
    parser.add_argument('--N_test', type=int, default=100)
    parser.add_argument('--matches_per_sample', type=int, default=25)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size_train', type=int, default=100)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--dataset', choices=['static', 'dynamic', 'dynamic_beachball'], default='dynamic')
    parser.add_argument('--beachball_sigma_factors', type=lambda s: [float(item) for item in s.split(',')],
                        default=[0.1, 0.5, 2, 10],
                        help='Heteroscedastic point cloud that has different noise levels (resembling a beachball).')
    parser.add_argument('--max_rotation_angle', type=float, default=180.,
                        help='In degrees. Maximum axis-angle rotation of simulated rotation.')

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--enforce_psd', action='store_true', default=False)
    parser.add_argument('--unit_frob', action='store_true', default=False)

    args = parser.parse_args([])
    print(args)

    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.double if args.double else torch.float

    if args.dataset == 'static':
        train_data, test_data = create_experimental_data_fast(args.N_train, args.N_test, args.matches_per_sample,
                                                              sigma=args.sim_sigma, device=device, dtype=tensor_type)
    else:
        # Data will be generated on the fly
        train_data, test_data = None, None

    def A_vec_to_quat(A_vec):
        A = convert_Avec_to_A(A_vec)
        _, evs = torch.symeig(A, eigenvectors=True)
        return evs[:, :, 0].squeeze()


    class BinghamNetDemo(torch.nn.Module):
        def __init__(self, enforce_psd=True, unit_frob_norm=True, batchnorm=False):
            super(BinghamNetDemo, self).__init__()
            self.A_net = PointNet(dim_out=10, normalize_output=False, batchnorm=batchnorm)
            self.enforce_psd = enforce_psd
            # self.unit_frob_norm = unit_frob_norm
            # self.qcqp_solver = QuadQuatFastSolver.apply

        def output_A(self, x):
            A_vec = self.A_net(x)
            if self.enforce_psd:
                A_vec = convert_Avec_to_Avec_psd(A_vec)
            if self.unit_frob_norm:
                A_vec = normalize_Avec(A_vec)

            return convert_Avec_to_A(A_vec)

        def forward(self, x):
            A_vec = self.A_net(x)

            # if self.enforce_psd:
            #     A_vec = convert_Avec_to_Avec_psd(A_vec)
            # if self.unit_frob_norm:
            #     A_vec = normalize_Avec(A_vec)
            # q = self.qcqp_solver(A_vec)
            q = A_vec_to_quat(A_vec)
            return q

    model_quat = PointNet(dim_out=4, normalize_output=True).to(device=device, dtype=tensor_type)
    loss_fn_quat = quat_chordal_squared_loss

    model_6D = RotMat6DDirect().to(device=device, dtype=tensor_type)
    loss_fn_6D = rotmat_frob_squared_norm_loss

    model_bingham = QuatNet(enforce_psd=args.enforce_psd, unit_frob_norm=args.unit_frob).to(device=device,
                                                                                            dtype=tensor_type)
    loss_fn_bingham = quat_chordal_squared_loss

    models = [model_quat, model_6D, model_bingham]
    losses = [loss_fn_quat, loss_fn_6D, loss_fn_bingham]
    rotmat_targets = [False, True, False]
    (_, _) = train_test_models_with_plots(args, train_data, test_data, models, losses, rotmat_targets=rotmat_targets)
