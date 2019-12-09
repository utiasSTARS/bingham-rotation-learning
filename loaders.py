import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import os.path as osp
from PIL import Image
import os
from quaternions import rotmat_to_quat

class SevenScenesData(Dataset):
    def __init__(self, scene, data_path, train, transform=None, output_first_image=True):
        
        """
          :param scene: scene name: 'chess', 'pumpkin', ...
          :param data_path: root 7scenes data directory.

        """
        self.transform = transform
        self.train = train
          # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)   
          # decide which sequences to use
        if train:
            split_file = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
    
        # read poses and collect image names
        self.c_imgs = []
        self.pose_files = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
            frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten() for i in frame_idx]
            ps[seq] = np.asarray(pss)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx]
            self.c_imgs.extend(c_imgs)
        
        self.poses = np.empty((0,16))
        for seq in seqs:
            self.poses = np.vstack((self.poses,ps[seq]))

        self.poses = torch.from_numpy(self.poses).float()

        if output_first_image:
            self.first_image = self.transform(self.load_image(self.c_imgs[0]))
            self.C_w_c0 = self.poses[0].view(4,4)[:3, :3]
            print(self.C_w_c0)
        else:
            self.first_image = None

        print('Loaded {} poses'.format(self.poses.shape[0]))

    def __getitem__(self, index):
        img = self.transform(self.load_image(self.c_imgs[index]))
        pose = self.poses[index].view(4,4) #Poses are camera to world
        C_ci_w = pose[:3,:3].transpose(0,1) #World to camera
        
        if self.first_image is not None:
            return (self.first_image, img), rotmat_to_quat(C_ci_w.mm(self.C_w_c0))
        else:
            return img, rotmat_to_quat(C_ci_w)

    def __len__(self):
        return self.poses.shape[0]

    def load_image(self, filename, loader=default_loader):
        try:
            img = loader(filename)
        except IOError as e:
            print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
            return None
        except:
            print('Could not load image {:s}, unexpected error'.format(filename))
            return None
        return img

