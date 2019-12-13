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
class CustomResNetDirect(torch.nn.Module):
    def __init__(self, dual=True):
        super(CustomResNetDirect, self).__init__()
        if dual:
            self.cnn = CustomResNetDual(num_outputs=4, normalize_output=True)
        else:
            self.cnn = CustomResNet(num_outputs=4, normalize_output=True)

    def forward(self, im):
        return self.cnn(im)

class CustomResNetConvex(torch.nn.Module):
    def __init__(self, dual=True):
        super(CustomResNetConvex, self).__init__()
        if dual:
            self.cnn = CustomResNetDual(num_outputs=10, normalize_output=True)
        else:
            self.cnn = CustomResNet(num_outputs=10, normalize_output=True)
        self.qcqp_solver = QuadQuatFastSolver.apply

    def forward(self, im):
        A_vec = self.cnn(im)
        q = self.qcqp_solver(A_vec)
        return q
   

class CustomResNetDual(torch.nn.Module):
    def __init__(self, num_outputs, normalize_output=True):
        super(CustomResNetDual, self).__init__()
        self.cnn = torchvision.models.resnet34(pretrained=True)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = torch.nn.Linear(num_ftrs, 512)
        self.head = torch.nn.Sequential(
          torch.nn.PReLU(),
          torch.nn.Linear(1024, 128),
          torch.nn.PReLU(),
          torch.nn.Linear(128, num_outputs)
        )
        self.normalize_output = normalize_output
        self.freeze_layers()

    def freeze_layers(self):
        # To freeze or not to freeze...
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.fc.parameters():
            param.requires_grad = True

    def forward(self, ims):
        feats = torch.cat((self.cnn(ims[0]), self.cnn(ims[1])), dim=1)
        y = self.head(feats)
        if self.normalize_output:
            y = y/y.norm(dim=1).view(-1, 1)
        return y


class CustomResNet(torch.nn.Module):
    def __init__(self, num_outputs, normalize_output=True):
        super(CustomResNet, self).__init__()
        self.cnn = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = torch.nn.Linear(num_ftrs, num_outputs)
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
