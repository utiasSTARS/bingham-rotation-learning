import torch
import time, argparse
from datetime import datetime
import numpy as np
import sys
sys.path.insert(0,'..')
from loaders import FLADataset
from networks import *
from losses import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tqdm
from helpers_train_test import train_test_model


def main():
    parser = argparse.ArgumentParser(description='Autoencoder on FLA data')
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--batch_size_test', type=int, default=64)
    parser.add_argument('--batch_size_train', type=int, default=32)

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--megalith', action='store_true', default=False)
    parser.add_argument('--scene', choices=['indoor', 'outdoor'], default='outdoor')
    parser.add_argument('--save_model', action='store_true', default=False)


    parser.add_argument('--dim_latent', type=int, default=16)
    parser.add_argument('--dim_transition', type=int, default=128)

    parser.add_argument('--lr', type=float, default=5e-4)


    args = parser.parse_args()
    print(args)

    #Float or Double?
    tensor_type = torch.float


    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    tensor_type = torch.float


    #Monolith
    if args.megalith:
        dataset_dir = '/media/datasets/'
    else:
        dataset_dir = '/media/m2-drive/datasets/'

    image_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/flea3'
    pose_dir = dataset_dir+'fla/2020.01.14_rss2020_data/2017_05_10_10_18_40_fla-19/pose'

    normalize = transforms.Normalize(mean=[0.45],
                                    std=[0.25])

    transform = transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    # model = ComplexAutoEncoder(dim_in=1, dim_latent=args.dim_latent, dim_transition=args.dim_transition).to(device=device, dtype=tensor_type)
    # print(model)
    # return

    #test_dataset = '../experiments/FLA/{}_test.csv'.format(args.scene)
    train_dataset = '../experiments/FLA/{}_train_reverse_False.csv'.format(args.scene)

    train_loader = DataLoader(FLADataset(train_dataset, image_dir=image_dir, pose_dir=pose_dir, transform=transform),
                            batch_size=args.batch_size_train, pin_memory=False,
                            shuffle=True, num_workers=args.num_workers, drop_last=False)

    
    #model = ConvAutoencoder().to(device=device, dtype=tensor_type)
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
        saved_data_file_name = 'fla_autoencoder_model_{}_{}'.format(args.scene, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
        full_saved_path = '../saved_data/fla/{}.pt'.format(saved_data_file_name)
        torch.save({
                'train_dataset': train_dataset,
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
