import torch
import numpy as np
import torch.nn.functional as F
from convex_layers import *
from utils import sixdim_to_rotmat
import torchvision


class RotMat6DDirect(torch.nn.Module):
    def __init__(self):
        super(RotMat6DDirect, self).__init__()        
        self.net = PointNet(dim_out=6, normalize_output=False)

    def forward(self, x):
        vecs = self.net(x)
        C = sixdim_to_rotmat(vecs)
        return C

class QuatNet(torch.nn.Module):
    def __init__(self, enforce_psd=True, unit_frob_norm=True):
        super(QuatNet, self).__init__()
        self.A_net = PointNet(dim_out=10, normalize_output=False)
        self.enforce_psd = enforce_psd
        self.unit_frob_norm = unit_frob_norm
        self.qcqp_solver = QuadQuatFastSolver.apply
    
    def output_A(self, x):
        A_vec = self.A_net(x)
        if self.enforce_psd:
            A_vec = convert_Avec_to_Avec_psd(A_vec)
        if self.unit_frob_norm:
            A_vec = normalize_Avec(A_vec)
        
        return convert_Avec_to_A(A_vec)

    def forward(self, x):
        A_vec = self.A_net(x)

        if self.enforce_psd:
            A_vec = convert_Avec_to_Avec_psd(A_vec)
        if self.unit_frob_norm:
            A_vec = normalize_Avec(A_vec)
        
        q = self.qcqp_solver(A_vec)
        return q


class PointFeatCNN(torch.nn.Module):
    def __init__(self):
        super(PointFeatCNN, self).__init__()
        self.net = torch.nn.Sequential(
                torch.nn.Conv1d(6, 64, kernel_size=1),
                torch.nn.PReLU(),
                torch.nn.Conv1d(64, 128, kernel_size=1),
                torch.nn.PReLU(),
                torch.nn.Conv1d(128, 1024, kernel_size=1),
                torch.nn.AdaptiveMaxPool1d(output_size=1)
                )

    def forward(self, x):
        x = self.net(x)
        return x.squeeze()

class PointFeatMLP(torch.nn.Module):
    def __init__(self, num_pts):
        super(PointFeatMLP, self).__init__()

        self.num_pts = num_pts
        self.net = torch.nn.Sequential(
                torch.nn.Linear(3*num_pts, 3*num_pts),
                #torch.nn.BatchNorm1d(128),
                torch.nn.PReLU(),
                torch.nn.Linear(3*num_pts, 1024),
                #torch.nn.BatchNorm1d(128),
                torch.nn.PReLU(),
                torch.nn.Linear(1024, 512),
                )

    def forward(self, x):
        x = self.net(x)
        return x

        
class PointNet(torch.nn.Module):
    def __init__(self, dim_out=10, normalize_output=False):
        super(PointNet, self).__init__()
        self.feat_net = PointFeatCNN()
        self.normalize_output = normalize_output
        self.head = torch.nn.Sequential(
          torch.nn.Linear(1024, 256),
          torch.nn.PReLU(),
          torch.nn.Linear(256, 128),
          torch.nn.PReLU(),
          torch.nn.Linear(128, dim_out)
        )

    def forward(self, x):

        #Decompose input into two point clouds
        if x.dim() < 4:
            x = x.unsqueeze(dim=0)

        x_1 = x[:, 0, :, :].transpose(1,2)
        x_2 = x[:, 1, :, :].transpose(1,2)


        feats_12 = self.feat_net(torch.cat([x_1, x_2], dim=1))

        if feats_12.dim() < 2:
            feats_12 = feats_12.unsqueeze(dim=0)
        
        out = self.head(feats_12)

        if self.normalize_output:
            out = out / out.norm(dim=1, keepdim=True)
        
        return out



#CNNS
class RotMat6DFlowNet(torch.nn.Module):
    def __init__(self):
        super(RotMat6DFlowNet, self).__init__()        
        self.net = BasicCNN(dim_in=2, dim_out=6, normalize_output=False)
    def forward(self, x):
        vecs = self.net(x)
        C = sixdim_to_rotmat(vecs)
        return C

class QuatFlowNet(torch.nn.Module):
    def __init__(self, enforce_psd=True, unit_frob_norm=True):
        super(QuatFlowNet, self).__init__()
        self.A_net = BasicCNN(dim_in=2, dim_out=10, normalize_output=False)
        self.enforce_psd = enforce_psd
        self.unit_frob_norm = unit_frob_norm
        self.qcqp_solver = QuadQuatFastSolver.apply
    
    def output_A(self, x):
        A_vec = self.A_net(x)
        if self.enforce_psd:
            A_vec = convert_Avec_to_Avec_psd(A_vec)
        if self.unit_frob_norm:
            A_vec = normalize_Avec(A_vec)
        
        return convert_Avec_to_A(A_vec)

    def forward(self, x):
        A_vec = self.A_net(x)

        if self.enforce_psd:
            A_vec = convert_Avec_to_Avec_psd(A_vec)
        if self.unit_frob_norm:
            A_vec = normalize_Avec(A_vec)
        
        q = self.qcqp_solver(A_vec)
        return q



def conv_unit(in_planes, out_planes, kernel_size=3, stride=2,padding=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(out_planes),
            torch.nn.ReLU()
        )


class BasicCNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, normalize_output=True):
        super(BasicCNN, self).__init__()
        self.normalize_output = normalize_output
        self.cnn = torch.nn.Sequential(
            conv_unit(dim_in, 64, kernel_size=3, stride=2, padding=1),
            conv_unit(64, 128, kernel_size=3, stride=2, padding=1),
            conv_unit(128, 256, kernel_size=3, stride=2, padding=1),
            conv_unit(256, 512, kernel_size=3, stride=2, padding=1),
            conv_unit(512, 1024, kernel_size=3, stride=2, padding=1),
            conv_unit(1024, 1024, kernel_size=3, stride=2, padding=1),
            conv_unit(1024, 1024, kernel_size=3, stride=2, padding=1)
        )
        self.fc = torch.nn.Linear(4096, dim_out)


    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        if self.normalize_output:
            out = out/out.norm(dim=1).view(-1, 1)
        return out


class CustomResNet(torch.nn.Module):
    def __init__(self, dim_out, normalize_output=True):
        super(CustomResNet, self).__init__()
        self.cnn = torchvision.models.resnet34(pretrained=True)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = torch.nn.Linear(num_ftrs, dim_out)
        self.normalize_output = normalize_output
        
    def forward(self, x):
        y = self.cnn(x)
        if self.normalize_output:
            y = y/y.norm(dim=1).view(-1, 1)
        return y

    def freeze_layers(self):
        # To freeze or not to freeze...
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Keep the FC layer active..
        for param in self.cnn.fc.parameters():
            param.requires_grad = True
