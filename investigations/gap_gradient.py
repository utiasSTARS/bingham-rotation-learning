import numpy as np
import torch
import sys
sys.path.insert(0,'..')
from convex_layers import QuadQuatFastSolver
from losses import quat_squared_loss, rotmat_frob_squared_norm_loss
from helpers_sim import create_experimental_data_fast
from networks import QuatNet, PointNet
from gen_uncertainty_plots import sum_bingham_dispersion_coeff, first_eig_gap
from quaternions import *
from utils import sixdim_to_rotmat

def test_single_grad():
    A = np.random.randn(4,4)
    A = A + A.T
    q_t = np.random.randn(4)
    q_t = q_t / np.linalg.norm(q_t)

    els, evs = np.linalg.eigh(A)
    B = els[0]*np.eye(4) - A
    B_pinv = np.linalg.pinv(B)

    s = evs[:,0].dot(np.kron(evs[:,0].reshape(1, 4), B_pinv))

    print(s)

    u_i = evs[:, [0]]


    b = q_t.reshape(1,4).dot(np.kron(evs[:,0].reshape(1, 4), B_pinv))
    print(b.dot(np.kron(u_i, u_i)))

def test_quat_norms():
    device = torch.device('cpu')
    tensor_type = torch.float64
    train_data, test_data = create_experimental_data_fast(50, 10, 100, sigma=0.05, device=device, dtype=tensor_type)
    model = PointNet(dim_out=4, normalize_output=False, batchnorm=False).to(device=device, dtype=tensor_type)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = quat_squared_loss
    model.train()
    for i in range(100):
        # Reset gradient
        optimizer.zero_grad()

        # Forward
        out = model.forward(train_data.x)
        q = out/out.norm(dim=1, keepdim=True)
        loss = loss_fn(q, train_data.q)
        
        x_test = test_data.x
        out_test = model.forward(x_test)

        x_rand = torch.rand_like(test_data.x)
        out_rand = model.forward(x_rand)
        
        # Backward
        loss.backward()

        # Update parameters
        optimizer.step()
        print('Iter: {}. Loss: {:.5F}. 4-Norm: {:.5F}. 4-Norm (Test): {:.5F}. 4-Norm (Random): {:.5F}'.format(i, loss.item(), out.norm(dim=1).mean(), out_test.norm(dim=1).mean(), out_rand.norm(dim=1).mean()))

def test_rotmat_to_quat():

    
def test_6D_norms():
    device = torch.device('cpu')
    tensor_type = torch.float64
    train_data, test_data = create_experimental_data_fast(1, 1, 100, sigma=0.01, device=device, dtype=tensor_type)
    model = PointNet(dim_out=6, normalize_output=False, batchnorm=False).to(device=device, dtype=tensor_type)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    loss_fn = rotmat_frob_squared_norm_loss
    model.train()
    for i in range(100):
        # Reset gradient
        optimizer.zero_grad()

        # Forward
        out = model.forward(train_data.x)
        C = sixdim_to_rotmat(out)
        loss = loss_fn(C.squeeze(), quat_to_rotmat(train_data.q))
        #x_test = torch.rand_like(test_data.x)
        
        x_test = test_data.x
        
        out_test = model.forward(x_test)

        # Backward
        loss.backward()

        # Update parameters
        optimizer.step()
        print('Iter: {}. Loss: {:.5F}. 6D Norm: {:.5F}. 6D Norm (Test): {:.5F}.'.format(i, loss.item(), out.norm(), out_test.norm()))


def test_gaps_network():
    loss_fn = quat_squared_loss
    device = torch.device('cpu')
    tensor_type = torch.float64
    train_data, test_data = create_experimental_data_fast(1, 1, 100, sigma=0.01, device=device, dtype=tensor_type)
    model = QuatNet(enforce_psd=False, unit_frob_norm=False).to(device=device, dtype=tensor_type)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for i in range(1000):
        # Reset gradient
        optimizer.zero_grad()

        # Forward
        out = model.forward(train_data.x)
        
        A = model.output_A(train_data.x).detach().squeeze()
        

        el = np.linalg.eigvalsh(A.numpy())
        #print(el)
        # B = -A.numpy() + el[0]*np.eye(4)
        # #print('lambda_3: {:.5F}'.format(np.linalg.eigvalsh(B)[2]))

        # spacings = np.diff(el, axis=1)

        sum_coeff  = sum_bingham_dispersion_coeff(A.numpy())
        #B = -A.numpy() + el[0]*np.eye(4)
        #print(np.linalg.eigvalsh(B))
        #x_randn = torch.rand_like(train_data.x)
        x_randn = torch.rand_like(test_data.x)
        A_test = model.output_A(x_randn).detach().squeeze().numpy()
        # el = np.linalg.eigvalsh(A_test)
        # B = -A_test + el[0]*np.eye(4)

        #print('lambda_3 (test): {:.5F}'.format(np.linalg.eigvalsh(B)[2]))
        #print('lambda_3 (random): {:.5F}'.format(np.linalg.eigvalsh(B)[2]))

        sum_coeff_test  = sum_bingham_dispersion_coeff(A_test)
        # el = np.linalg.eigvalsh(A_test)
        # spacings_test = np.diff(el)


        loss = loss_fn(out, train_data.q)

        # Backward
        loss.backward()

        # Update parameters
        optimizer.step()
        print('Iter: {}. Loss: {:.5F}. Frob Norm: {:.5F} Mean SDC: {:.5F}. Mean SDC (test) {:.5F}.'.format(i, loss.item(),A.norm(dim=(0,1)),  sum_coeff.mean(), sum_coeff_test.mean()))

if __name__ == "__main__":
    #test_single_grad()
    #test_gaps_network()
    #test_6D_norms()
    test_quat_norms()