from quaternions import *
from networks import *
from sim_helpers import *
from liegroups.numpy import SO3
import torch

path = 'saved_data/synthetic/synthetic_wahba_experiment_12-10-2019-17-17-32.pt'
checkpoint = torch.load(path)
print('Data loaded')
args = checkpoint['args']
A_net = ANet(num_pts=args.matches_per_sample, bidirectional=args.bidirectional_loss).double()
model = QuatNet(A_net=A_net)
model.load_state_dict(checkpoint['model_rep'], strict=False)

train_data, test_data = create_experimental_data(10, 100, args.matches_per_sample, sigma=args.sim_sigma)

