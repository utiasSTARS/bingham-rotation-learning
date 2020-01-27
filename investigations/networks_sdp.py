import torch
import numpy as np
import torch.nn.functional as F
from convex_layers import *
from utils import sixdim_to_rotmat
import torchvision
from sdp_layers import RotMatSDPSolver


class RotMatSDPNet(torch.nn.Module):
    def __init__(self, dim_rep=55, enforce_psd=True, unit_frob_norm=True, batchnorm=True):
        super(RotMatSDPNet, self).__init__()        
        self.net = PointNet(dim_out=dim_rep, normalize_output=False, batchnorm=batchnorm)
        self.rotation_layer = RotMatSDPSolver()
        self.enforce_psd = enforce_psd
        self.unit_frob_norm = unit_frob_norm

    def forward(self, x):
        A_vec = self.net(x)
        if A_vec.shape[-1] == 55:
            if self.enforce_psd:
                A_vec = convert_Avec_to_Avec_psd(A_vec)
            if self.unit_frob_norm:
                A_vec = normalize_Avec(A_vec)

        C = self.rotation_layer(A_vec)
        return C

