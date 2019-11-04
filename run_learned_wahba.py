import torch
from torch.autograd import gradcheck
import numpy as np
from nets_and_solvers import ANet, QuadQuatSolver
from helpers import quat_norm_diff, gen_sim_data
from liegroups.torch import SO3
import time

class ExperimentalData():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


#Generic training function
def train_minibatch(model, loss_fn, optimizer, x, targets):
    #Ensure model gradients are active
    model.train()

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    loss = loss_fn(model.forward(x), targets)

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    return loss.item() 

def test_model(model, loss_fn, x, targets):
    #model.eval() speeds things up because it turns off gradient computation
    model.eval()
    y = model.forward(x)
    loss = loss_fn(y, targets)
    return (y, loss)


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


def create_experimental_data(N_train=500, N_test=50, N_matches_per_sample=10):

    x_train = torch.zeros(N_train, N_matches_per_sample*2*3)
    x_test = torch.zeros(N_test, N_matches_per_sample*2*3)
    y_train = torch.zeros(N_train, 4)
    y_test = torch.zeros(N_test, 4)
    
    for n in range(N_train):
        # sample_ids = np.random.choice(x_1.shape[0], N_matches_per_sample, replace=False)
        # sample_ids = torch.from_numpy(sample_ids)
        C, x_1, x_2 = gen_sim_data(N=N_matches_per_sample, sigma=0.01, torch_vars=True)
        q = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        x_train[n, :] = torch.cat([x_1.flatten(), x_2.flatten()])
        y_train[n, :] = q

    for n in range(N_test):
        C, x_1, x_2 = gen_sim_data(N=N_matches_per_sample, sigma=0.01, torch_vars=True)
        q = SO3.from_matrix(C).to_quaternion(ordering='xyzw')
        x_test[n, :] = torch.cat([x_1.flatten(), x_2.flatten()])
        y_test[n, :] = q
            
    return ExperimentalData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def main():
    
    #Sim parameters
    N_train = 1000
    N_test = 50
    N_matches_per_sample = 10

    #Learning Parameters
    num_epochs = 100
    batch_size = 10

    torch.manual_seed(42)
    model = ANet(num_pts=N_matches_per_sample)
    loss_fn = quat_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    exp_data = create_experimental_data(N_train, N_test, N_matches_per_sample)
    N_train = exp_data.x_train.shape[0]
    N_test = exp_data.x_test.shape[0]


    for e in range(num_epochs):
        start_time = time.time()

        #Train model
        print('Training...')
        num_batches = N_train // batch_size
        train_loss = torch.tensor(0.)
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            train_loss += (1/num_batches)*train_minibatch(model, loss_fn, optimizer, exp_data.x_train[start:end], exp_data.y_train[start:end])
        
        #Test model
        print('Testing...')
        num_batches = N_test // batch_size
        test_loss = torch.tensor(0.)
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            (y_test, test_loss_k) = test_model(model, loss_fn, exp_data.x_test[start:end], exp_data.y_test[start:end])
            test_loss += (1/num_batches)*test_loss_k

        elapsed_time = time.time() - start_time

        #test_angle = quat_to_angle_metric(torch.sqrt(2*test_loss))
        #train_angle = quat_to_angle_metric(torch.sqrt(2*train_loss))
        
        test_angle = quat_to_angle_metric(test_loss)
        train_angle = quat_to_angle_metric(train_loss)


        print('Epoch: {}/{}. Train Loss {:.3f} (deg) | Test Loss: {:.3f} (deg). Epoch time: {:.3f} sec.'.format(e+1, num_epochs, train_angle, test_angle, elapsed_time))


if __name__=='__main__':
    main()
