import torch
import numpy as np
import torch.nn.functional as F
from convex_layers import *
from utils import sixdim_to_rotmat
import torchvision


class RotMat6DDirect(torch.nn.Module):
    def __init__(self, batchnorm=False):
        super(RotMat6DDirect, self).__init__()        
        self.net = PointNet(dim_out=6, normalize_output=False, batchnorm=batchnorm)

    def forward(self, x):
        vecs = self.net(x)
        C = sixdim_to_rotmat(vecs)
        return C

class QuatNet(torch.nn.Module):
    def __init__(self, enforce_psd=True, unit_frob_norm=True, batchnorm=False):
        super(QuatNet, self).__init__()
        self.A_net = PointNet(dim_out=10, normalize_output=False, batchnorm=batchnorm)
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
    def __init__(self, batchnorm=False):
        super(PointFeatCNN, self).__init__()
        if batchnorm:
            self.net = torch.nn.Sequential(
                    torch.nn.Conv1d(6, 64, kernel_size=1),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.PReLU(),
                    torch.nn.Conv1d(64, 128, kernel_size=1),
                    torch.nn.BatchNorm1d(128),
                    torch.nn.PReLU(),
                    torch.nn.Conv1d(128, 1024, kernel_size=1),
                    torch.nn.AdaptiveMaxPool1d(output_size=1)
            )
        else:
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

class PointNetInspect(torch.nn.Module):
    def __init__(self, dim_out=10, normalize_output=False, batchnorm=False):
        super(PointNetInspect, self).__init__()
        self.feat_net = PointFeatCNN(batchnorm=batchnorm)
        self.normalize_output = normalize_output
        self.head = torch.nn.Sequential(
          torch.nn.Linear(1024, 256),
          torch.nn.PReLU(),
          torch.nn.Linear(256, 128),
          torch.nn.PReLU()
        )
        self.final_layer = torch.nn.Linear(128, dim_out, bias=True)

    def pre_forward(self, x):
        #Decompose input into two point clouds
        if x.dim() < 4:
            x = x.unsqueeze(dim=0)

        x_1 = x[:, 0, :, :].transpose(1,2)
        x_2 = x[:, 1, :, :].transpose(1,2)


        feats_12 = self.feat_net(torch.cat([x_1, x_2], dim=1))

        if feats_12.dim() < 2:
            feats_12 = feats_12.unsqueeze(dim=0)
        
        out = self.head(feats_12)
        #out = out / out.sum(dim=1, keepdim=True)
        return out

    def forward(self, x):
        #weights = self.pre_forward(x)
        #inv_norms = 1./self.final_layer.weight.norm(dim=0, keepdim=True)
        #out = self.final_layer(weights*inv_norms)
        
        out = self.final_layer(self.pre_forward(x))
        
        if self.normalize_output:
            out = out / out.norm(dim=1, keepdim=True)
        
        return out
        
class PointNet(torch.nn.Module):
    def __init__(self, dim_out=10, normalize_output=False, batchnorm=False):
        super(PointNet, self).__init__()
        self.feat_net = PointFeatCNN(batchnorm=batchnorm)
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
    def __init__(self, dim_in=2, batchnorm=True):
        super(RotMat6DFlowNet, self).__init__()        
        self.net = BasicCNN(dim_in=dim_in, dim_out=6, normalize_output=False, batchnorm=batchnorm)
    def forward(self, x):
        vecs = self.net(x)
        C = sixdim_to_rotmat(vecs)
        return C

class RotMatSDPFlowNet(torch.nn.Module):
    def __init__(self, dim_in=2, dim_rep=55, enforce_psd=True, unit_frob_norm=True, batchnorm=True):
        super(RotMatSDPFlowNet, self).__init__()        
        self.net = BasicCNN(dim_in=dim_in, dim_out=dim_rep, normalize_output=False, batchnorm=batchnorm)
        self.sdp_solver = RotMatSDPSolver()
        self.enforce_psd = enforce_psd
        self.unit_frob_norm = unit_frob_norm

    def forward(self, x):
        A_vec = self.net(x)
        if A_vec.shape[-1] == 55:
            if self.enforce_psd:
                A_vec = convert_Avec_to_Avec_psd(A_vec)
            if self.unit_frob_norm:
                A_vec = normalize_Avec(A_vec)
        C = self.sdp_solver(A_vec)
        return C

class QuatFlowNet(torch.nn.Module):
    def __init__(self, enforce_psd=True, unit_frob_norm=True, dim_in=2, batchnorm=True):
        super(QuatFlowNet, self).__init__()
        self.A_net = BasicCNN(dim_in=dim_in, dim_out=10, normalize_output=False, batchnorm=batchnorm)
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

class QuatFlowResNet(torch.nn.Module):
    def __init__(self, enforce_psd=True, unit_frob_norm=True):
        super(QuatFlowResNet, self).__init__()
        self.A_net = CustomResNet(dim_out=10)
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

def conv_unit(in_planes, out_planes, kernel_size=3, stride=2,padding=1, batchnorm=True):
        if batchnorm:
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
                torch.nn.BatchNorm2d(out_planes),
                torch.nn.PReLU()
            )
        else:
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
                torch.nn.PReLU()
            )



class BasicCNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, normalize_output=True, batchnorm=True):
        super(BasicCNN, self).__init__()
        self.normalize_output = normalize_output
        self.cnn = torch.nn.Sequential(
            conv_unit(dim_in, 64, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(64, 128, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(128, 256, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(256, 512, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(512, 1024, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(1024, 1024, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(1024, 1024, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm)
        )
        self.fc = torch.nn.Sequential(
                    torch.nn.Linear(4096, 512),
                    torch.nn.PReLU(),
                    torch.nn.Linear(512, dim_out)
        )
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        if self.normalize_output:
            out = out/out.norm(dim=1).view(-1, 1)
        return out



def deconv_unit(in_planes, out_planes, kernel_size=3, stride=2, padding=1, batchnorm=True):
    if batchnorm:
        return torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(out_planes),
            torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        )
    else:
        return torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        )
class BasicAutoEncoder(torch.nn.Module):
    def __init__(self, dim_in, dim_latent, dim_transition, normalize_output=True, batchnorm=True):
        super(BasicAutoEncoder, self).__init__()
        self.normalize_output = normalize_output
        self.cnn = torch.nn.Sequential(
            conv_unit(dim_in, 64, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(64, 128, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(128, 256, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(256, 512, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(512, 1024, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(1024, 1024, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            conv_unit(1024, 1024, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm)
        )
        self.fc_encoder = torch.nn.Sequential(
            torch.nn.Linear(4096, dim_transition),
            torch.nn.PReLU(),
            torch.nn.Linear(dim_transition, dim_latent),
        )
        self.fc_decoder = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Linear(dim_latent, dim_transition),
            torch.nn.PReLU(),
            torch.nn.Linear(dim_transition, 4096)
        )
        self.cnn_decode = torch.nn.Sequential(
            deconv_unit(1024, 1024, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            deconv_unit(1024, 1024, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            deconv_unit(1024, 512, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            deconv_unit(512, 256, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            deconv_unit(256, 128, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            deconv_unit(128, 64, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm),
            deconv_unit(64, dim_in, kernel_size=3, stride=2, padding=1, batchnorm=batchnorm)
        )

    def encode(self, x):
        code = self.cnn(x)
        code = code.view(code.shape[0], -1)
        code = self.fc_encoder(code)
        if self.normalize_output:
            code = code/code.norm(dim=1).view(-1, 1)
        return code

    def decode(self, x):
        out = self.fc_decoder(x)
        out = self.cnn_decode(out)
        return out

    def forward(self, x):
        code = self.encode(x)
        out = self.decode(code)
        # if self.normalize_output:
        #     out = out / out.norm(dim=1).view(-1, 1)
        return out, code


class CustomResNet(torch.nn.Module):
    def __init__(self, dim_out, normalize_output=False):
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

