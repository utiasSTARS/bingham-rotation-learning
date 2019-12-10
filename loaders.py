import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import os.path as osp
from PIL import Image
import os
from quaternions import rotmat_to_quat
#import cv2

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


        # self.extractor = cv2.ORB()
        # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if output_first_image:
            self.first_image = self.transform(self.load_image(self.c_imgs[0]))
            self.C_w_c0 = self.poses[0].view(4,4)[:3, :3]

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


    # def match_to_first_image(self, im):
    #     kp2, des2 = self.extractor.detectAndCompute(im1,None)
    #     matches = bf.match(des1,des2)
    #     matches = sorted(matches, key = lambda x:x.distance)

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


class KITTIVODatasetPreTransformed(Dataset):
    """KITTI Odometry Benchmark dataset with full memory read-ins."""

    def __init__(self, kitti_dataset_file, seqs_base_path, transform_img=None, run_type='train', use_flow=True, apply_blur=False, reverse_images=False, seq_prefix='seq_', use_only_seq=None):
        self.kitti_dataset_file = kitti_dataset_file
        self.seqs_base_path = seqs_base_path
        self.apply_blur = apply_blur
        self.transform_img = transform_img
        self.seq_prefix = seq_prefix
        self.load_kitti_data(run_type, use_only_seq)  # Loads self.image_quad_paths and self.labels
        self.use_flow = use_flow
        self.reverse_images = reverse_images

    def load_kitti_data(self, run_type, use_only_seq):
        with open(self.kitti_dataset_file, 'rb') as handle:
            kitti_data = pickle.load(handle)

        if run_type == 'train':
            self.seqs = kitti_data['train_seqs']
            self.pose_indices = kitti_data['train_pose_indices']
            self.T_21_gt = kitti_data['train_T_21_gt']
            self.T_21_vo = kitti_data['train_T_21_vo']
            self.pose_deltas = kitti_data['train_pose_deltas']

        elif run_type == 'test':
            self.seqs = kitti_data['test_seqs']
            self.pose_indices = kitti_data['test_pose_indices']
            self.T_21_gt = kitti_data['test_T_21_gt']
            self.T_21_vo = kitti_data['test_T_21_vo']
            self.pose_delta = kitti_data['test_pose_delta']

        else:
            raise ValueError('run_type must be set to `train`, or `test`. ')

        if use_only_seq is not None:
            self.pose_indices = [self.pose_indices[i] for i in range(len(self.seqs))
                                 if self.seqs[i] ==  use_only_seq]
            self.T_21_gt = [self.T_21_gt[i] for i in range(len(self.seqs))
                                 if self.seqs[i] == use_only_seq]
            self.T_21_vo = [self.T_21_vo[i] for i in range(len(self.seqs))
                                 if self.seqs[i] == use_only_seq]
            self.seqs = [self.seqs[i] for i in range(len(self.seqs))
                                 if self.seqs[i] == use_only_seq]

        print('Loading sequences...{}'.format(list(set(self.seqs))))
        print('Pose delta: {}'.format(self.pose_indices[0][1] - self.pose_indices[0][0]))
        self.seq_images = {seq: self.import_seq(seq) for seq in list(set(self.seqs))}
        print('...done loading images into memory.')

    def import_seq(self, seq):
        file_path = self.seqs_base_path + '/' + self.seq_prefix + '{}.pt'.format(seq)
        data = torch.load(file_path)
        return data['im_l']

    def __len__(self):
        return len(self.T_21_gt)

    def prep_img(self, img):
        if self.transform_img is not None:
            return self.transform_img(img.float()/255.)
        else:
            return img.float() / 255.

    def compute_flow(self, img1, img2, idx, apply_blur = False):
        #Convert back to W x H x C
        np_img1 = cv2.cvtColor(img1.permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY)
        np_img2 = cv2.cvtColor(img2.permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY)

        if apply_blur:
            np_img1 = cv2.GaussianBlur(np_img1, (13, 13), 0)
            np_img2 = cv2.GaussianBlur(np_img2, (13, 13), 0)

        flow_cv2 = cv2.calcOpticalFlowFarneback(np_img1, np_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_img = torch.from_numpy(flow_cv2).permute(2,0,1)

        # if idx < 10:
        #     # Obtain the flow magnitude and direction angle
        #     hsvImg = np.zeros_like(img1.permute(1,2,0).numpy())
        #     hsvImg[..., 1] = 255
        #     mag, ang = cv2.cartToPolar(flow_cv2[..., 0], flow_cv2[..., 1])
        #     # Update the color image
        #     hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
        #     hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        #     rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
        #     cv2.imwrite('{}_flow.png'.format(idx), rgbImg)
        #gr_img1 = torch.from_numpy(np_img1).float().unsqueeze(0)
        #gr_img2 = torch.from_numpy(np_img2).float().unsqueeze(0)

        #stacked_img = torch.cat((gr_img1, gr_img2, flow_img), 0)
        return flow_img


    def __getitem__(self, idx):
        seq = self.seqs[idx]
        p_ids = self.pose_indices[idx]
        C_21_gt = self.T_21_gt[idx].rot.as_matrix()


        if self.reverse_images:
            p_ids = [p_ids[1], p_ids[0]]
            C_21_gt = self.T_21_gt[idx].rot.inv().as_matrix()

        #print('Loading seq: {}. ids: {}'.format(seq, p_ids))


        #C_21_err = self.T_21_gt[idx].rot.as_matrix().dot(self.T_21_vo[idx].rot.as_matrix().transpose())

        # image_pair = [self.prep_img(self.seq_images[seq][p_ids[0]]),
        #               self.prep_img(self.seq_images[seq][p_ids[1]])]
        if self.use_flow:
            img_input = self.compute_flow(self.seq_images[seq][p_ids[0]], self.seq_images[seq][p_ids[1]], idx, self.apply_blur)
        else:
            img_input = [self.prep_img(self.seq_images[seq][p_ids[0]]),
                       self.prep_img(self.seq_images[seq][p_ids[1]])]

        q_target = torch.from_numpy(quaternion_from_matrix(C_21_gt)).float()
        return img_input, q_target
