import torch
import time, argparse
from datetime import datetime
import numpy as np
import sys
sys.path.insert(0,'..')
from loaders import KITTIVODatasetPreTransformed
from networks import *
from losses import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tqdm
from helpers_train_test import train_test_model


def main():
    parser = argparse.ArgumentParser(description='Autoencoder on KITTI data')
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--batch_size_test', type=int, default=64)
    parser.add_argument('--batch_size_train', type=int, default=32)

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--megalith', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)


    parser.add_argument('--dim_latent', type=int, default=16)
    parser.add_argument('--dim_transition', type=int, default=128)
    parser.add_argument('--seq', choices=['00', '02', '05'], default='00')

    parser.add_argument('--lr', type=float, default=5e-4)


    args = parser.parse_args()
    print(args)

    #Float or Double?
    tensor_type = torch.float
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')


    transform = None
    seqs_base_path = '/media/m2-drive/datasets/KITTI/single_files'
    if args.megalith:
        seqs_base_path = '/media/datasets/KITTI/single_files'

    seq_prefix = 'seq_'

    kitti_data_pickle_file = '../experiments/kitti/kitti_singlefile_data_sequence_{}_delta_1_reverse_False_min_turn_0.0.pickle'.format(args.seq)
    
    train_loader = DataLoader(KITTIVODatasetPreTransformed(kitti_data_pickle_file, use_flow=False, seqs_base_path=seqs_base_path, transform_img=transform, run_type='train', seq_prefix=seq_prefix),
                            batch_size=args.batch_size_train, pin_memory=False,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)

    model = ComplexAutoEncoder(dim_in=1, dim_latent=args.dim_latent, dim_transition=args.dim_transition).to(device=device, dtype=tensor_type)

    loss_fn = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for e in range(args.epochs):
        start_time = time.time()

        #Train model
        model.train()
        train_loss = torch.tensor(0.)
        num_train_batches = len(train_loader)

        pbar = tqdm.tqdm(total=num_train_batches)
        for _, (imgs, _) in enumerate(train_loader):
            #Move all data to appropriate device
            img = imgs[:,[0],:,:].to(device=device, dtype=tensor_type)
            _, train_loss_k = train_autoenc(model, loss_fn, optimizer, img)
            
            train_loss += (1./num_train_batches)*train_loss_k
            pbar.update(1)
        
        pbar.close()
        elapsed_time = time.time() - start_time
        
        output_string = 'Epoch: {}/{}. Train: Loss {:.3E}. Epoch time: {:.3f} sec.'.format(e+1, args.epochs, train_loss, elapsed_time)
        print(output_string)
    
    if args.save_model:
        saved_data_file_name = 'kitti_autoencoder_seq_{}_{}'.format(args.seq, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = '../saved_data/kitti/{}.pt'.format(saved_data_file_name)
        torch.save({
                'kitti_data_pickle_file': kitti_data_pickle_file,
                'model': model.state_dict(),
                'args': args,
            }, full_saved_path)
        print('Saved data to {}.'.format(full_saved_path))

#Generic training function
def train_autoenc(model, loss_fn, optimizer, img):

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    img_out, code = model.forward(img)
    
    loss = loss_fn(img_out, img)
    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    return (img_out, loss.item())


def test_autoenc(model, loss_fn, img):
    # Forward
    with torch.no_grad():
        img_out, code = model.forward(img)
        loss = loss_fn(img_out, img)
            
    return (img_out, loss.item())

if __name__=='__main__':
    main()
