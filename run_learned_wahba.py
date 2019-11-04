import torch
import time
import numpy as np
from liegroups.torch import SO3
from nets_and_solvers import ANet, QuadQuatSolver
from convex_wahba import build_A
from helpers import quat_norm_diff, gen_sim_data, solve_horn

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
    loss = loss_fn(model.forward(x, A_prior), targets)

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    return loss.item() 

def test_model(model, loss_fn, x, targets, A_prior=None):
    #model.eval() speeds things up because it turns off gradient computation
    model.eval()
    # Forward
    q = model.forward(x, A_prior)
    loss = loss_fn(q, targets)
    return (q, loss)


def quat_loss(q_in, q_target):
    d = quat_norm_diff(q_in, q_target)
    losses =  d
    return losses.mean()

#See Rotation Averaging by Hartley et al. (2013)
def quat_to_angle_metric(q_met, units='deg'):
    angle = 4.*torch.asin(0.5*q_met)
    if units == 'deg':
        angle = (180./np.pi)*angle
    elif units == 'rad':
        pass
    else:
        raise RuntimeError('Unknown units in metric conversion.')
    return angle


def create_experimental_data(N_train=2000, N_test=50, N_matches_per_sample=100):

    x_train = torch.zeros(N_train, N_matches_per_sample*2*3)
    q_train = torch.zeros(N_train, 4)
    A_prior_train = torch.zeros(N_train, 4, 4)

    x_test = torch.zeros(N_test, N_matches_per_sample*2*3)
    q_test = torch.zeros(N_test, 4)
    A_prior_test = torch.zeros(N_test, 4, 4)

    sigma_prior = 0.01
    sigma_sim_vec = sigma_prior*np.ones(N_matches_per_sample)
    sigma_sim_vec[:int(N_matches_per_sample/2)] *= 10.
    sigma_prior_vec = 0.01*np.ones(N_matches_per_sample)
    

    for n in range(N_train):

        C, x_1, x_2 = gen_sim_data(N_matches_per_sample, sigma_sim_vec, torch_vars=True)
        q = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        x_train[n] = torch.cat([x_1.flatten(), x_2.flatten()])
        q_train[n] = q
        A_prior_train[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))

    for n in range(N_test):
        C, x_1, x_2 = gen_sim_data(N_matches_per_sample, sigma_sim_vec, torch_vars=True)
        q = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        x_test[n] = torch.cat([x_1.flatten(), x_2.flatten()])
        q_test[n] = q
        A_prior_test[n] = torch.from_numpy(build_A(x_1.numpy(), x_2.numpy(), sigma_2=sigma_prior_vec**2))
    
    train_data = SyntheticData(x_train, q_train, A_prior_train)
    test_data = SyntheticData(x_test, q_test, A_prior_test)
    
    return train_data, test_data


def compute_mean_horn_error(data):
    N = data.x.shape[0]
    err = torch.empty(N)
    for i in range(N):
        x = data.x[i]
        x_1, x_2 = torch.chunk(x, 2)
        x_1 = x_1.view(-1, 3).numpy()
        x_2 = x_2.view(-1, 3).numpy()
        C = torch.from_numpy(solve_horn(x_1, x_2))
        q_est = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        err[i] = quat_loss(q_est, data.q[i])
    return quat_to_angle_metric(err.mean())

def main():
    
    #Sim parameters
    N_train = 1000
    N_test = 100
    N_matches_per_sample = 100

    #Learning Parameters
    num_epochs = 100
    batch_size = 10
    use_A_prior = True


    torch.manual_seed(42)
    model = ANet(num_pts=N_matches_per_sample)
    loss_fn = quat_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    train_data, test_data = create_experimental_data(N_train, N_test, N_matches_per_sample)

    print('Generated training data...')
    print('Mean Horn error | Train (deg): {:.3f}. Test: {:.3f} (deg).'.format(compute_mean_horn_error(train_data), compute_mean_horn_error(test_data)))

    N_train = train_data.x.shape[0]
    N_test = test_data.x.shape[0]

    for e in range(num_epochs):
        start_time = time.time()

        #Train model
        print('Training...')
        num_batches = N_train // batch_size
        train_loss = torch.tensor(0.)
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size

            if use_A_prior:
                A_prior = train_data.A_prior[start:end]
            else:
                A_prior = None
            train_loss_k = train_minibatch(model, loss_fn, optimizer, train_data.x[start:end], train_data.q[start:end], A_prior=A_prior)
            train_loss += (1/num_batches)*train_loss_k
        
        #Test model
        print('Testing...')
        num_batches = N_test // batch_size
        test_loss = torch.tensor(0.)
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            if use_A_prior:
                A_prior = test_data.A_prior[start:end]
            else:
                A_prior = None
            (q_test, test_loss_k) = test_model(model, loss_fn, test_data.x[start:end], test_data.q[start:end], A_prior=A_prior)
            test_loss += (1/num_batches)*test_loss_k

        elapsed_time = time.time() - start_time

        #test_angle = quat_to_angle_metric(torch.sqrt(2*test_loss))
        #train_angle = quat_to_angle_metric(torch.sqrt(2*train_loss))
        
        test_angle = quat_to_angle_metric(test_loss)
        train_angle = quat_to_angle_metric(train_loss)


        print('Epoch: {}/{}. Train Loss {:.3f} (deg) | Test Loss: {:.3f} (deg). Epoch time: {:.3f} sec.'.format(e+1, num_epochs, train_angle, test_angle, elapsed_time))


if __name__=='__main__':
    main()
