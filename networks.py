import torch
import numpy as np
import torch.nn.functional as F
from convex_layers import QuadQuatFastSolver
import torchvision 

#Utility module to replace BatchNorms without affecting structure
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class QuatNetDirect(torch.nn.Module):
    def __init__(self, num_pts):
        super(QuatNetDirect, self).__init__()        
        self.net = ANet(num_pts=num_pts, num_dim_out=4)

    def forward(self, x, A_prior=None):
        vecs = self.net(x)
        q = vecs/vecs.norm(dim=1).view(-1, 1)
        return q


class QuatNet(torch.nn.Module):
    def __init__(self, A_net=None):
        super(QuatNet, self).__init__()
        if A_net is None:
            raise RuntimeError('Must pass in an ANet to QuatNet')
        self.A_net = A_net
        self.qcqp_solver = QuadQuatFastSolver.apply

    def forward(self, x, A_prior=None):
        A_vec = self.A_net(x, A_prior)
        if self.A_net.bidirectional:
            q = self.qcqp_solver(A_vec[0])
            q_inv = self.qcqp_solver(A_vec[1])
            return [q, q_inv]
        else:
            q = self.qcqp_solver(A_vec)
            return q


class APriorNet(torch.nn.Module):
    def __init__(self):
        super(APriorNet, self).__init__()
        self.fc1 = torch.nn.Linear(10,10)
        self.bn1 = torch.nn.BatchNorm1d(10)

    def forward(self, A_vec):
        A_vec = F.relu(self.bn1(self.fc1(A_vec))) + A_vec
        return A_vec

class PointFeatCNN(torch.nn.Module):
    def __init__(self):
        super(PointFeatCNN, self).__init__()
        self.net = torch.nn.Sequential(
                torch.nn.Conv1d(3, 64, kernel_size=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(64, 128, kernel_size=1),
                torch.nn.LeakyReLU(),
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
        return x.squeeze()

        
class ANet(torch.nn.Module):
    def __init__(self, num_pts, num_dim_out=10, bidirectional=False):
        super(ANet, self).__init__()
        self.num_pts = num_pts
        self.bidirectional = bidirectional #Evaluate both forward and backward directions
        self.A_prior_net = APriorNet()
        self.feat_net1 = PointFeatMLP(num_pts=num_pts)
        self.feat_net2 = PointFeatMLP(num_pts=num_pts)
        

        self.head = torch.nn.Sequential(
          torch.nn.Linear(1024, 256),
          #torch.nn.BatchNorm1d(128),
          torch.nn.PReLU(),
          torch.nn.Linear(256, 128),
          #torch.nn.BatchNorm1d(128),
          torch.nn.PReLU(),
          torch.nn.Linear(128, num_dim_out)
        )


    def feats_to_A(self, x):
        A_vec = self.head(x)
        A_vec = A_vec/A_vec.norm(dim=1).view(-1, 1)

        return A_vec

    def forward(self, x, A_prior=None):
        #Decompose input into two point clouds
        # x_1 = x[:, 0, :, :].transpose(1,2)
        # x_2 = x[:, 1, :, :].transpose(1,2)

        x_1 = x[:, 0, :, :].view(-1, self.num_pts*3)
        x_2 = x[:, 1, :, :].view(-1, self.num_pts*3)

        #Collect and concatenate features
        #x_1 -> x_2
        feats_12 = torch.cat([self.feat_net1(x_1), self.feat_net2(x_2)], dim=1)
        #feats_12 = self.feat_net1(x_2)

        A1 = self.feats_to_A(feats_12)
        
        #Prior? Doesn't make sense with symmetric loss unless we give two priors...TODO
        # if A_prior is not None:
        #     A1 = A1 + self.A_prior_net(A_prior)

        if self.bidirectional:
            #x_2 -> x_1
            feats_21 = torch.cat([self.feat_net1(x_2), self.feat_net2(x_1)], dim=1)
            A2 = self.feats_to_A(feats_21)
            return [A1, A2]

        return A1

class ANetSingle(torch.nn.Module):
    def __init__(self, num_pts, bidirectional=False):
        super(ANetSingle, self).__init__()
        self.num_pts = num_pts
        self.bidirectional = bidirectional

        self.body = torch.nn.Sequential(
          torch.nn.Linear(num_pts*3, 512),
          #torch.nn.BatchNorm1d(128),
          torch.nn.ELU(),
          torch.nn.Linear(512, 256),
          #torch.nn.BatchNorm1d(128),
          torch.nn.ELU(),
          torch.nn.Linear(256, 128),
          #torch.nn.BatchNorm1d(64),
          torch.nn.ELU(),
          torch.nn.Linear(128, 10)
        )



    def forward(self, x, A_prior=None):
        #Decompose input into two point clouds
        #x_1 = x[:, 0, :, :].transpose(1,2)
        x_2 = x[:, 1, :, :].view(-1, self.num_pts*3)

        A_vec = self.body(x_2)
        A_vec = A_vec/A_vec.norm(dim=1).view(-1, 1)
        return A_vec



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
        self.cnn = torchvision.models.resnet34(pretrained=True)
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
