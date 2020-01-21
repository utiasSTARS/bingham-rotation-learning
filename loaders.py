import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torchvision
import os.path as osp
from PIL import Image
import os
from quaternions import rotmat_to_quat, quat_to_rotmat, rotmat_angle_diff
from liegroups.torch import SO3
import pickle
import cv2
import torch.utils.data as tud


class KITTIVODatasetPreTransformed(Dataset):
    """KITTI Odometry Benchmark dataset with full memory read-ins."""

    def __init__(self, kitti_dataset_file, seqs_base_path, output_sample_images=0, 
    transform_img=None, transform_second_half_only=False, run_type='train', 
    use_flow=True, apply_blur=False, reverse_images=False, seq_prefix='seq_', 
    use_only_seq=None, rotmat_targets=False):

        self.kitti_dataset_file = kitti_dataset_file
        self.seqs_base_path = seqs_base_path
        self.apply_blur = apply_blur
        self.transform_img = transform_img
        self.transform_second_half_only = transform_second_half_only

        self.seq_prefix = seq_prefix
        self.load_kitti_data(run_type, use_only_seq)  # Loads self.image_quad_paths and self.labels
        self.use_flow = use_flow
        self.reverse_images = reverse_images
        self.rotmat_targets = rotmat_targets
        
        #Output for visualization
        self.output_sample_images = output_sample_images
        if self.output_sample_images > 0:
            self.output_image_idx = np.random.choice(len(self.T_21_gt), self.output_sample_images, replace=False)
            print('Will output image at idx:')
            print(self.output_image_idx)
        else:
            self.output_image_idx = []

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
            self.T_21_gt = [torch.from_numpy(self.T_21_gt[i]).float() for i in range(len(self.seqs))
                                 if self.seqs[i] == use_only_seq]
            self.T_21_vo = [torch.from_numpy(self.T_21_vo[i]).float() for i in range(len(self.seqs))
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

        return flow_img


    def __getitem__(self, idx):
        seq = self.seqs[idx]
        p_ids = self.pose_indices[idx]
        C_21_gt = self.T_21_gt[idx][:3,:3]

        if self.reverse_images:
            p_ids = [p_ids[1], p_ids[0]]
            C_21_gt = self.T_21_gt[idx][:3,:3].transpose(0,1)

        if self.use_flow:
            img_input = self.compute_flow(self.seq_images[seq][p_ids[0]], self.seq_images[seq][p_ids[1]], idx, self.apply_blur)
        else:
            #Should we transform?
            transform_img_flag = False
            if self.transform_img is not None:
                if self.transform_second_half_only:
                    if idx > len(self.T_21_gt)/2:
                        transform_img_flag = True
                else:
                    transform_img_flag = True

            if transform_img_flag:
                img_input = torch.cat([self.transform_img(self.prep_img(self.seq_images[seq][p_ids[0]])),
                            self.transform_img(self.prep_img(self.seq_images[seq][p_ids[1]]))], dim=0)
            else:
                img_input = torch.cat([self.prep_img(self.seq_images[seq][p_ids[0]]),
                            self.prep_img(self.seq_images[seq][p_ids[1]])], dim=0)

        if idx in self.output_image_idx:
            file_name = 'img_{0}.png'.format(idx)
            print('Saving....{}'.format(file_name))
            torchvision.utils.save_image(img_input[:3], file_name)
    
        if self.rotmat_targets:
            return img_input, torch.from_numpy(C_21_gt).float()
        else:
            return img_input, rotmat_to_quat(torch.from_numpy(C_21_gt).float())


def pointnet_collate(batch):
    data = torch.cat([item[0] for item in batch], dim=0)
    target = torch.cat([item[1] for item in batch], dim=0)
    return [data, target]

class PointNetDataset(Dataset):
    """PointNet Dataset."""

    def __init__(self, pc_folder, rotations_per_batch=50, 
    total_iters=1e6, 
    dtype=torch.float, 
    rotmat_targets=False,
    load_into_memory=True, 
    device=torch.device('cpu'),
    test_mode=False):
        """
        Args:

        """
        self.file_list = self._load_pc_list(pc_folder)
        self.total_iters = int(total_iters)
        self.rotations_per_batch = rotations_per_batch
        self.dtype = dtype
        self.rotmat_targets = rotmat_targets
        self.test_mode = test_mode
        if load_into_memory:
            print('Loading pointclouds into memory...')
            self.data = [torch.from_numpy(np.array(self._load_file(file))) for file in self.file_list]
            print('Done')
        else:
            self.data = None
    
    # See: https://github.com/papagina/RotationContinuity
    def _load_pc_list(self, d):
        files = [os.path.join(d, f) for f in os.listdir(d)] 
        return files

    def _load_file(self, path):
        """takes as input the path to a .pts and returns a list of 
        tuples of floats containing the points in in the form:
        [(x_0, y_0, z_0),
        (x_1, y_1, z_1),
        ...
        (x_n, y_n, z_n)]"""
        with open(path) as f:
            rows = [rows.strip() for rows in f]
        
        """Use the curly braces to find the start and end of the point data""" 
        #head = rows.index('{') + 1
        #tail = rows.index('}')

        """Select the point data split into coordinates"""
        raw_points = rows#rows[head:tail]
        coords_set = [point.split() for point in raw_points]

        """Convert entries from lists of strings to tuples of floats"""
        points = [tuple([float(point) for point in coords]) for coords in coords_set]
        return (points)


    def __len__(self):
        if self.test_mode:
            return len(self.file_list)
        else:
            return self.total_iters

    def __getitem__(self, idx):
        # Select a random point cloud
        if self.test_mode:
            pointcloud_id = idx
        else:
            pointcloud_id = torch.randint(len(self.file_list), (1,)).item() 
        
        if self.data is None:
            pc1 = torch.from_numpy(np.array(self._load_file(self.file_list[pointcloud_id])))
        else:
            pc1 = self.data[pointcloud_id]

        #Matches the original code
        point_num = int(pc1.shape[0]/2)
        #Sub sample
        pc1 = pc1[:point_num]
        
        batch_num = self.rotations_per_batch
        pc1 = pc1.view(1, point_num,3).expand(batch_num,point_num,3).transpose(1,2) #batch*3*p_num
        C = SO3.exp(torch.randn(batch_num, 3, dtype=torch.double)).as_matrix()

        pc2 = torch.bmm(C, pc1) #(batch*point_num)*3*1

        x = torch.empty(batch_num, 2, point_num, 3)
        x[:,0,:,:] = pc1.transpose(1,2)
        x[:,1,:,:] = pc2.transpose(1,2)

        if self.rotmat_targets:
            targets = C
        else:
            targets = rotmat_to_quat(C, ordering='xyzw')
        targets = targets.to(self.dtype)
        x = x.to(self.dtype)
        
        return (x, targets)


class SevenScenesData(Dataset):
    def __init__(self, scene, data_path, train, transform=None, output_first_image=True, tensor_type=torch.float):
        
        """
          :param scene: scene name: 'chess', 'pumpkin', ...
          :param data_path: root 7scenes data directory.

        """
        self.transform = transform
        self.train = train
        self.tensor_type = tensor_type
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

        self.poses = torch.from_numpy(self.poses).to(dtype=tensor_type)

        if output_first_image:
            self.first_image = self.transform(self.load_image(self.c_imgs[0])).to(dtype=tensor_type)
            self.C_w_c0 = self.poses[0].view(4,4)[:3, :3]

        else:
            self.first_image = None


        print('Loaded {} poses'.format(self.poses.shape[0]))

    def __getitem__(self, index):
        img = self.transform(self.load_image(self.c_imgs[index])).to(dtype=self.tensor_type)
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


class FLADataset(tud.Dataset):
    """Loads FLA data from ASL format into a torch dataset.
    """

    def __init__(self, image_dir, pose_dir, select_idx=None, transform=None, rotmat_targets=False, eval_mode=False):
        """Constructor for FLADataset.

        :param image_dir: Root directory of images.
        :param pose_dir: Root directory of poses.
        :param transform: Transform to apply when reading data.
        """
        self.image_dir = image_dir
        self.pose_dir = pose_dir
        self.transform = transform
        self.rotmat_targets = rotmat_targets
        self.eval_mode = train_mode

        # Read in images.
        self.image_timestamps = [] # nanoseconds.
        self.image_filenames = []
        with open(os.path.join(self.image_dir, "data.csv"), "r") as ff:
            lines = ff.readlines()
            lines = [line.rstrip() for line in lines] # Strip newlines.
            lines = [line for line in lines if line[0] is not "#"] # Strip comments.

            for line in lines:
                tokens = line.split(",")
                self.image_timestamps.append(np.uint64(tokens[0]))
                self.image_filenames.append(tokens[1])

        self.image_timestamps = np.array(self.image_timestamps)

        # Read poses.
        self.pose_timestamps = [] # nanoseconds.
        self.pose_qxyzw = []
        with open(os.path.join(self.pose_dir, "data.csv"), "r") as ff:
            lines = ff.readlines()
            lines = [line.rstrip() for line in lines] # Strip newlines.
            lines = [line for line in lines if line[0] is not "#"] # Strip comments.

            for line in lines:
                tokens = line.split(",")
                self.pose_timestamps.append(np.uint64(tokens[0]))

                qw = np.float(tokens[-4])
                qx = np.float(tokens[-3])
                qy = np.float(tokens[-2])
                qz = np.float(tokens[-1])

                self.pose_qxyzw.append(np.array([qx, qy, qz, qw]))

        self.pose_timestamps = np.array(self.pose_timestamps)
        self.pose_qxyzw = torch.from_numpy(np.array(self.pose_qxyzw))

        #Filter
        if select_idx is not None:

            self.image_filenames = self.image_filenames[select_idx[0]:select_idx[1]]
            self.image_timestamps = self.image_timestamps[select_idx[0]:select_idx[1]]
            # self.pose_timestamps = self.pose_timestamps[select_idx[0]:select_idx[1]]
            # self.pose_qxyzw = self.pose_qxyzw[select_idx[0]:select_idx[1]]

        return

    def __len__(self):
        if self.eval_mode:
            return len(self.image_filenames) - 2
        else:
            return len(self.image_filenames) - 2 + len(self.image_filenames) - 3
            
    
    def compute_flow(self, img1, img2):
        #Convert back to W x H x C
        np_img1 = img1.permute(1,2,0).numpy()
        np_img2 = img2.permute(1,2,0).numpy()

        flow_cv2 = cv2.calcOpticalFlowFarneback(np_img1, np_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_img = torch.from_numpy(flow_cv2).permute(2,0,1)

        return flow_img

    def find_pose(self, timestamp):
         # Find closest pose given timestamp.
        pose_idx = np.argmin(np.abs(self.pose_timestamps - timestamp))
        
        tol_ms = 30
        assert(np.abs(np.float(self.pose_timestamps[pose_idx]) - timestamp) * 1e-6 < tol_ms)
        return pose_idx

    def __getitem__(self, idx):
        
        if self.eval_mode:
            id1 = idx
            id2 = id1 + 1
        else:
            if idx < len(self.image_filenames):
                id1 = idx
                id2 = id1 + 1
            else:
                id1 = idx - len(self.image_filenames) 
                id2 = id1 + 2
            
        image1 = Image.open(os.path.join(self.image_dir, "data", self.image_filenames[id1]))
        image2 = Image.open(os.path.join(self.image_dir, "data", self.image_filenames[id2]))

        pose_idx1 = self.find_pose(self.image_timestamps[id1])
        pose_idx2 = self.find_pose(self.image_timestamps[id2])

        R_1 = quat_to_rotmat(self.pose_qxyzw[pose_idx1, :], ordering='xyzw')
        R_2 = quat_to_rotmat(self.pose_qxyzw[pose_idx2, :], ordering='xyzw')

        R = R_1.mm(R_2.transpose(0,1))
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        if self.rotmat_targets:
            target = R
        else:
            target = rotmat_to_quat(R, ordering='xyzw')

        #flow_image = self.compute_flow(image1, image2)
        img_input = torch.cat([image1, image2], dim=0)
        return img_input, target