import numpy as np
import torch
import sys
sys.path.insert(0,'..')
from helpers_sim import SyntheticData
from convex_layers import QuadQuatFastSolver
from losses import quat_squared_loss, rotmat_frob_squared_norm_loss
from networks import QuatNet, PointNet, PointNetInspect
from quaternions import *
from liegroups.torch import SO3 as SO3_torch


def create_experiment(N_train=100, N_test=10, N_matches_per_sample=50, sigma=0.01, angle_limits=[0,180.], dtype=torch.double, device=torch.device('cpu')):
    C_train, x_1_train, x_2_train = gen_sim_data(N_train, N_matches_per_sample, sigma, angle_limits=angle_limits)
    C_test, x_1_test, x_2_test = gen_sim_data(N_test, N_matches_per_sample, sigma, angle_limits=angle_limits)

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


def gen_sim_data(N_rotations, N_matches_per_rotation, sigma, angle_limits=[0, 180.], dtype=torch.double):
    ##Simulation
    #Create a random rotation
    axis = torch.randn(N_rotations, 3, dtype=dtype)
    axis = axis / axis.norm(dim=1, keepdim=True)
    
    fac = (np.pi/180.)
    angle = fac*(angle_limits[1] - angle_limits[0])*torch.rand(N_rotations, 1) + fac*angle_limits[0]

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



def test_discontinuity_unit_quat():
    tensor_type = torch.float64
    angle_limits = [140.,160.]
    train_data, test_data = create_experiment(N_train=10, N_test=10, sigma=0.01, angle_limits=angle_limits, dtype=tensor_type)
    model = PointNetInspect(dim_out=4, normalize_output=False, batchnorm=False).to(dtype=tensor_type)
    #optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = quat_squared_loss
    model.train()
    for i in range(200):
        # Reset gradient
        optimizer.zero_grad()

        # Forward
        model.train()
        out = model.forward(train_data.x)
        q = out/out.norm(dim=1, keepdim=True)
        loss = loss_fn(q, train_data.q)
        mean_err = quat_angle_diff(q, train_data.q)


        # Backward
        loss.backward()

        # Update parameters
        optimizer.step()
        
        #Print unnormalized quaternions
        #print(out)
        
        #Final layer weights
        # model.final_layer.weight
        # model.final_layer.bias

        #Test data
        # model.eval()
        # x_test = test_data.x
        # out_test = model.forward(x_test)

        #Penultimate outputs ('alpha_i's)
        #alpha = model.pre_forward(x_test)

        print('Epoch {}. Loss: {:.3F}. Mean angle err: {:.3F}'.format(i, loss.item(), mean_err))
    
    

if __name__ == "__main__":
    test_discontinuity_unit_quat()