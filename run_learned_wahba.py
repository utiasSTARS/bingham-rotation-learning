import torch
from torch.autograd import gradcheck
import numpy as np
from nets_and_solvers import ANetwork, QuadQuatSolver
from helpers import quat_norm_diff, gen_sim_data
from liegroups.numpy import SO3

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
    losses = 0.5*d*d
    return losses.mean()

def create_experimental_data(N_train=2000, N_test=100, N_matches_per_sample=10):

    C, x_1, x_2 = gen_sim_data(N=100, sigma=0.001, torch_vars=True)
    q = torch.from_numpy(SO3.from_matrix(C.numpy()).to_quaternion(ordering='xyzw'))

    x_train = torch.empty(N_train, N_matches_per_sample*2*3)
    x_test = torch.empty(N_train, N_matches_per_sample*2*3)
    y_train = torch.empty(N_train, 4)
    y_test = torch.empty(N_train, 4)
    
    for n in range(N_train):
        sample_ids = np.random.choice(x_1.shape[0], N_matches_per_sample, replace=False)
        sample_ids = torch.from_numpy(sample_ids)

        x_train[n, :] = torch.cat([x_1[sample_ids].flatten(), x_2[sample_ids].flatten()])
        y_train[n, :] = q

    for n in range(N_test):
        sample_ids = np.random.choice(x_1.shape[0], N_matches_per_sample, replace=False)
        sample_ids = torch.from_numpy(sample_ids)

        x_train[n, :] = torch.cat([x_1[sample_ids].flatten(), x_2[sample_ids].flatten()])
        y_train[n, :] = q
        
    return ExperimentalData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def main():
    #Parameters
    num_epochs = 100
    batch_size = 1

    torch.manual_seed(42)
    model = ANetwork(num_inputs=60, num_outputs=16)
    loss_fn = quat_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    exp_data = create_experimental_data()

    N_train = exp_data.x_train.shape[0]
    N_test = exp_data.x_test.shape[0]
    

    for e in range(num_epochs):
        #Train model
        num_batches = N_train // batch_size
        train_loss = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            train_loss += (1/num_batches)*train_minibatch(model, loss_fn, optimizer, exp_data.x_train[start:end], exp_data.y_train[start:end])
        
        #Test model
        num_batches = N_test // batch_size
        test_loss = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            (_, test_loss_k) = test_model(model, loss_fn, exp_data.x_test[start:end], exp_data.y_test[start:end])
            test_loss += (1/num_batches)*test_loss_k
        
        if e%1 == 0:
            print('Epoch: {}/{}. Train Loss {:.5E} | Test Loss: {:.5E}.'.format(e+1, num_epochs, train_loss, test_loss))

if __name__=='__main__':
    main()