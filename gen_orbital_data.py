from pyslam.sensors import StereoCamera
from liegroups.numpy import SE3, SO3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio


sim_setup = {
    "num_pts": 36, #Must be a square - i.e., 16,25, etc. (quirk of the way I generate a meshgrid)
    "num_poses": [15000, 500],  #Both num_poses and traj_names can be lists that correspond to different datasets
    "traj_names": ['train_abs', 'valid_abs_ood'],
    "absolute_pose": True,
    "pixel_noise_var": [1., 1.],
    "pixel_bias": [0., 0.],
    'semisphere_radius': 25,
    'angle_limits': [((-np.pi/3, np.pi/3), (-np.pi/4, np.pi/4),(-np.pi/4, np.pi/4)),
        ((-0.45*np.pi, 0.45*np.pi), (-np.pi/9,  np.pi/9),(-np.pi/9, np.pi/9))],
    'sequential_x': [False, True],
    'camera_intrinsics': {'w': 500, 'h': 500, 'cu': 250., 'cv': 250., 'f': 500, 'b': 1.0},
    "data_output_folder": "./orbital"
}


def normalize(x):
    return x / np.linalg.norm(x)



def create_semisphere(radius, num_pts):
    #Create a semi-sphere of points 'evenly' spaced on a semi-sphere (evenly in terms of angular metrics)
    #Outputs 'grids' that can be used with matplotlib's plot_surface

    #Th: Polar
    #Phi: Azimuth
    #For a semi-sphere: th: 0-90, phi: 0-360
    th = np.linspace(0., np.pi/2., np.sqrt(num_pts))
    phi = np.linspace(0., 2.*np.pi, np.sqrt(num_pts))
    th_grid, phi_grid = np.meshgrid(th, phi)

    x = radius*np.sin(th_grid)*np.cos(phi_grid)
    y = radius*np.sin(th_grid)*np.sin(phi_grid)
    z = radius*np.cos(th_grid)

    return (x,y,z)

def gen_points(num_pts, semisphere_radius):
    #Generate a a grid of points around the origin

    xlims = np.linspace(-0.3*semisphere_radius, 0.3*semisphere_radius, np.sqrt(num_pts))
    ylims = np.linspace(-0.3*semisphere_radius, 0.3*semisphere_radius, np.sqrt(num_pts))
    x, y = np.meshgrid(xlims, ylims)
    z = 0.*x

    #Return Nx3 matrix
    return np.hstack((x.reshape(num_pts, 1), y.reshape(num_pts, 1), z.reshape(num_pts, 1)))


def visualize_world(pts, T_vi_list, file_name, radius):
    #Output figure of the world
    fig = plt.figure()
    ax = Axes3D(fig)
    r_vi_i = np.empty((len(T_vi_list), 3))
    for i in range(len(T_vi_list)):
        r_vi_i[i] = T_vi_list[i].inv().trans

    (x,y,z) = create_semisphere(radius, 400)
        
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label='Landmarks', s=2.)
    ax.plot_surface(x, y, z, cmap=plt.cm.coolwarm, alpha=0.25)
    ax.scatter(r_vi_i[:, 0], r_vi_i[:, 1], r_vi_i[:, 2], '-*c', label='Trajectory')

    fig.savefig(file_name, bbox_inches='tight')
    return


#Sample poses from a semi sphere that can see all the landmarks
def create_orbital_poses(num_poses, semisphere_radius, angle_limits, sequential_x):
    T_vi_list = []
    (x_lim, y_lim, z_lim) = angle_limits

    if sequential_x:
        angle_x = np.linspace(x_lim[0], x_lim[1], num_poses)
    else:
        angle_x = (x_lim[1] - x_lim[0]) * np.random.rand(num_poses) + x_lim[0]

    angle_y = (y_lim[1] - y_lim[0]) * np.random.rand(num_poses) + y_lim[0]
    angle_z =  (z_lim[1] - z_lim[0])*np.random.rand(num_poses) + z_lim[0]


    for i in range(num_poses):

        C_vi = SO3.rotz(angle_z[i]).dot(SO3.roty(angle_y[i])).dot(SO3.rotx(angle_x[i]))
        r = np.array([0, 0, semisphere_radius])

        C_flip = SO3.rotx(np.pi)
        T1 = SE3(rot=C_vi, trans=np.zeros(3))
        T2 = SE3(rot=SO3.identity(), trans=-r)
        T3 = SE3(rot=C_flip, trans=np.zeros(3))

        T = T3.dot(T2).dot(T1)

        T_vi_list.append(T)

    return T_vi_list



def project_points(T_vi_gt, points, camera, pixel_noise_var):
    # Project world landmars into the camera frame and create a list of observations

    obs = []
    num_visible_pts = np.zeros(len(T_vi_gt))
    all_visible_pixels = []
    for pose_i, T in enumerate(T_vi_gt):
        #Project points with noise
        cam_pts = camera.project(T.dot(points))
        noisy_cam_pts = cam_pts + np.sqrt(pixel_noise_var)*np.random.randn(cam_pts.shape[0], 3)

        #Prune non-visible points and save
        pts_mask = camera.is_valid_measurement(noisy_cam_pts)
        num_visible_pts[pose_i] = np.sum(pts_mask)

        visible_pixels = noisy_cam_pts[pts_mask, :2]
        visible_ids = np.flatnonzero(pts_mask)
        uvd = noisy_cam_pts
        uvd[~pts_mask] = -1.0

        obs.append((visible_pixels, visible_ids, uvd)) 
        all_visible_pixels.append(visible_pixels)
    
    all_visible_pixels = np.concatenate(all_visible_pixels)
    print('Done projection')
    print('Mean u,v: {}, min u,v: {}, max u,v: {}. std u,v: {}.'.format(np.mean(all_visible_pixels, axis=0), np.min(all_visible_pixels, axis=0), np.max(all_visible_pixels, axis=0), np.std(all_visible_pixels, axis=0)))
    print('Number of visible points; mean {}, min {}, max {}.'.format(np.mean(num_visible_pts), np.min(num_visible_pts), np.max(num_visible_pts)))

    return obs


def save_mat_data(T_vi_gt, pts_w, obs, sim_setup, data_filename, t_i):
    # Save all sim data to a .mat file with the same variable names as AER1513 Assignment 3

    if sim_setup['absolute_pose']:
        # Ground truth vectors
        T_vk_i = np.empty((len(T_vi_gt), 4, 4))
        for k in range(len(T_vi_gt)):
            T_vk_i[k] = T_vi_gt[k].as_matrix()


        # Observations
        # Need 3 x num_poses x num_pts matrix
        y_k_j = np.empty((2, len(T_vi_gt), pts_w.shape[0]))

        for o_i, (_,_, uvd) in enumerate(obs):
            y_k_j[:, o_i, :] = uvd[:, :2].T
    else:
        T_vk_i = np.empty((round(len(T_vi_gt)/2), 4, 4))


        # Observations
        # Need 3 x num_poses x num_pts matrix
        y_k_j = np.empty((2, round(len(T_vi_gt)/2), pts_w.shape[0]))

        for o_i in range(0, len(obs), 2):
            _,_, uvd_1 = obs[o_i]
            _, _,uvd_2 = obs[o_i+1]

            unseen_pixels = uvd_1[:, 0] < 0
            obs_i = uvd_1[:, :2].T - uvd_2[:, :2].T + sim_setup['pixel_bias'][t_i]
            obs_i[:, unseen_pixels] = -1.
            y_k_j[:, round(o_i/2), :] = obs_i

            T_vk_i[round(o_i/2)] = T_vi_gt[o_i+1].dot(T_vi_gt[o_i].inv()).as_matrix()

    mat_dict = {}
    mat_dict['rho_i_pj_i'] = pts_w.T
    mat_dict['y_k_j'] = y_k_j
    mat_dict['y_var'] = np.ones((3,1))*sim_setup['pixel_noise_var']
    mat_dict['T_vk_i'] = T_vk_i
    mat_dict['cu'] = sim_setup['camera_intrinsics']['cu']
    mat_dict['cv'] = sim_setup['camera_intrinsics']['cv']
    mat_dict['fu'] = sim_setup['camera_intrinsics']['f'] 
    mat_dict['fv'] = sim_setup['camera_intrinsics']['f']
    mat_dict['b'] = sim_setup['camera_intrinsics']['b']

    sio.savemat(data_filename, mat_dict)
    


# Reproducibility
# np.random.seed(42)

# Landmarks (world frame is first camera frame)
# Let's create a bunch of them!
num_pts = sim_setup['num_pts']
pts_w = gen_points(num_pts, sim_setup['semisphere_radius'])
# Camera

camera = StereoCamera(
    sim_setup['camera_intrinsics']['cu'], 
    sim_setup['camera_intrinsics']['cv'],
    sim_setup['camera_intrinsics']['f'],
    sim_setup['camera_intrinsics']['f'],
    sim_setup['camera_intrinsics']['b'],
    sim_setup['camera_intrinsics']['w'],
    sim_setup['camera_intrinsics']['h'])

# Simulate a trajectory 
for t_i, traj in enumerate(sim_setup['traj_names']):
    print('Simulating {} data.'.format(traj))

    num_poses  = sim_setup['num_poses'][t_i]
    output_image = "{}/{}.png".format(sim_setup['data_output_folder'],traj)
    if sim_setup["absolute_pose"]:
        T_cw_list = create_orbital_poses(num_poses, sim_setup['semisphere_radius'], sim_setup['angle_limits'][t_i], sim_setup['sequential_x'][t_i])
    else:
        print("Creating small odometry with 6x1 observation vectors")
        T_cw_list = create_rand_odometry(num_poses, sim_setup['semisphere_radius'], sim_setup['angle_limits'][t_i])

    print('Done trajectory generation.')
    print('Points: {}. Poses: {}. Pixel Noise (Var.): {:.2f}'.format(num_pts, num_poses, sim_setup['pixel_noise_var'][t_i]))
    obs = project_points(T_cw_list, pts_w, camera, sim_setup['pixel_noise_var'][t_i])
    print('Visualizing world. Saving figure to: {}.'.format(output_image))
    visualize_world(pts_w, T_cw_list, output_image, sim_setup['semisphere_radius'])

    data_filename = "{}/{}.mat".format(sim_setup['data_output_folder'],traj)
    print('Saving data to: {}'.format(data_filename))
    save_mat_data(T_cw_list, pts_w, obs, sim_setup, data_filename, t_i)

    