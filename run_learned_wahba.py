import torch
import time
import numpy as np
from liegroups.torch import SO3
from nets_and_solvers import ANet, QuatNet
from convex_wahba import build_A
from helpers import quat_norm_diff, gen_sim_data, solve_horn, quat_inv

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
    out = model.forward(x, **kwargs)
    loss = loss_fn(out, targets)

    return (out, loss.item())

#See Rotation Averaging by Hartley et al. (2013)
def quat_norm_to_angle(q_met, units='deg'):
    angle = 4.*torch.asin(0.5*q_met)
    if units == 'deg':
        angle = (180./np.pi)*angle
    elif units == 'rad':
        pass
    else:
        raise RuntimeError('Unknown units in metric conversion.')
    return angle

def quat_consistency_loss(qs, q_target, reduce=True):
    q = qs[0]
    q_inv = qs[1]
    assert(q.shape == q_inv.shape == q_target.shape)
    d1 = quat_loss(q, q_target, reduce=False)
    d2 = quat_loss(q_inv, quat_inv(q_target), reduce=False)
    d3 = quat_loss(q, quat_inv(q_inv), reduce=False)
    losses =  d1*d1 + d2*d2 + d3*d3
    loss = losses.mean() if reduce else losses
    return loss
    

def quat_squared_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses =  0.5*d*d
    loss = losses.mean() if reduce else losses
    return loss

def quat_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses = d
    loss = losses.mean() if reduce else losses
    return loss

def quat_angle_diff(q, q_target, units='deg', reduce=True):
    assert(q.shape == q_target.shape)
    diffs = quat_norm_to_angle(quat_norm_diff(q, q_target), units=units)
    return diffs.mean() if reduce else diffs



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

        C, x_1, x_2 = gen_sim_data(N_matches_per_sample, sigma_sim_vec, torch_vars=True, shuffle_points=False)
        q = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        x_train[n, 0, :, :] = x_1
        x_train[n, 1, :, :] = x_2
        q_train[n] = q
        A_prior_train[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))

    for n in range(N_test):
        C, x_1, x_2 = gen_sim_data(N_matches_per_sample, sigma_sim_vec, torch_vars=True, shuffle_points=False)
        q = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        x_test[n, 0, :, :] = x_1
        x_test[n, 1, :, :] = x_2
        q_test[n] = q
        A_prior_test[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))
    
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
        A = A.unsqueeze()
    idx = torch.triu_indices(4,4)
    A_vec = A[:, idx[0], idx[1]]

    return A_vec.squeeze()


def pretrain(A_net, train_data, test_data):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(A_net.parameters(), lr=1e-3)
    batch_size = 100
    num_epochs = 500

    print('Pre-training A network...')
    N_train = train_data.x.shape[0]
    N_test = test_data.x.shape[0]
    num_batches = N_train // batch_size
    for e in range(num_epochs):
        start_time = time.time()

        #Train model
        train_loss = torch.tensor(0.)
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            _, train_loss_k = train_minibatch(A_net, loss_fn, optimizer,  train_data.x[start:end], convert_A_to_Avec(train_data.A_prior[start:end]))
            train_loss += (1/num_batches)*train_loss_k
    
        elapsed_time = time.time() - start_time

        #Test model
        num_batches = N_test // batch_size
        test_loss = torch.tensor(0.)
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            _, test_loss_k = test_model(A_net, loss_fn, test_data.x[start:end], convert_A_to_Avec(test_data.A_prior[start:end]))
            test_loss += (1/num_batches)*test_loss_k

        print('Epoch: {}/{}. Train: Loss {:.3E} | Test: Loss {:.3E}. Epoch time: {:.3f} sec.'.format(e+1, num_epochs, train_loss, test_loss, elapsed_time))

    return

def main():
    
    #Sim parameters
    sigma = 0.01
    N_train = 1000
    N_test = 100
    N_matches_per_sample = 10

    #Learning Parameters
    num_epochs = 100
    batch_size = 100
    use_A_prior = False #Only meaningful with symmetric_loss=False
    bidirectional_loss = False
    pretrain_A_net = False


    train_data, test_data = create_experimental_data(N_train, N_test, N_matches_per_sample, sigma=sigma)
    print('Generated training data...')
    print('Mean Horn Error. Train (deg): {:.3f} | Test: {:.3f} (deg).'.format(compute_mean_horn_error(train_data), compute_mean_horn_error(test_data)))


    A_net = ANet(num_pts=N_matches_per_sample, bidirectional=bidirectional_loss).double()
    if pretrain_A_net:
        pretrain(A_net, train_data, test_data)

    model = QuatNet(A_net=A_net)

    if bidirectional_loss:
        loss_fn = quat_consistency_loss
    else:
        loss_fn = quat_squared_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    N_train = train_data.x.shape[0]
    N_test = test_data.x.shape[0]

    for e in range(num_epochs):
        start_time = time.time()

        #Train model
        print('Training...')
        num_batches = N_train // batch_size
        train_loss = torch.tensor(0.)
        train_mean_err = torch.tensor(0.)
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size

            if use_A_prior:
                A_prior = convert_A_to_Avec(train_data.A_prior[start:end])
            else:
                A_prior = None
            
            (q_est, train_loss_k) = train_minibatch(model, loss_fn, optimizer, train_data.x[start:end], train_data.q[start:end], A_prior=A_prior)
            q_train = q_est[0] if bidirectional_loss else q_est
            train_loss += (1/num_batches)*train_loss_k
            train_mean_err += (1/num_batches)*quat_angle_diff(q_train, train_data.q[start:end])
        
        #Test model
        print('Testing...')
        num_batches = N_test // batch_size
        test_loss = torch.tensor(0.)
        test_mean_err = torch.tensor(0.)


        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            if use_A_prior:
                A_prior = convert_A_to_Avec(test_data.A_prior[start:end])
            else:
                A_prior = None
            (q_est, test_loss_k) = test_model(model, loss_fn, test_data.x[start:end], test_data.q[start:end], A_prior=A_prior)
            q_test = q_est[0] if bidirectional_loss else q_est
            test_loss += (1/num_batches)*test_loss_k
            test_mean_err += (1/num_batches)*quat_angle_diff(q_test, test_data.q[start:end])

        elapsed_time = time.time() - start_time


        print('Epoch: {}/{}. Train: Loss {:.3E} / Error {:.3f} (deg) | Test: Loss {:.3E} / Error {:.3f} (deg). Epoch time: {:.3f} sec.'.format(e+1, num_epochs, train_loss, train_mean_err, test_loss, test_mean_err, elapsed_time))

        A_pred = model.A_net.forward(train_data.x[[start]])
        A_pp = convert_A_to_Avec(train_data.A_prior[[start]])

if __name__=='__main__':
    main()
