import argparse
import os
import sys
from typing import List, Optional, Type, Union
import numpy as np
import math
import random
import torch
from torch_scatter import scatter_mean
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
import habitat
from habitat import logger
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from habitat_baselines.utils.env_utils import make_env_fn
from habitat_baselines.common.environments import get_env_class
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff
from habitat.tasks.utils import cartesian_to_polar

from vlnce_baselines.config.default import get_config


class ProjectorUtils():
    def __init__(self,
                vfov,
                batch_size,
                feature_map_height,
                feature_map_width,
                output_height,
                output_width,
                gridcellsize,
                world_shift_origin,
                z_clip_threshold,
                device,
                ):
            
        self.vfov = vfov
        self.batch_size = batch_size
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.output_height = output_height # dimensions of the topdown map
        self.output_width = output_width
        self.gridcellsize = gridcellsize
        self.z_clip_threshold = z_clip_threshold
        self.world_shift_origin = world_shift_origin
        self.device = device

        self.x_scale, self.y_scale, self.ones = self.compute_scaling_params(
            batch_size, feature_map_height, feature_map_width
        )

    
    def compute_intrinsic_matrix(self, width, height, vfov):
        hfov = width/height * vfov
        f_x = width / (2.0*math.tan(hfov/2.0))
        f_y = height / (2.0*math.tan(vfov/2.0))
        cy = height / 2.0
        cx = width / 2.0
        K = torch.Tensor([[f_x, 0, cx],
                          [0, f_y, cy],
                          [0, 0, 1.0]])
        return K


    def compute_scaling_params(self, batch_size, image_height, image_width):
        """ Precomputes tensors for calculating depth to point cloud """
        # (float tensor N,3,3) : Camera intrinsics matrix
        K = self.compute_intrinsic_matrix(image_width, image_height, self.vfov)
        K = K.to(device=self.device).unsqueeze(0)
        K = K.expand(batch_size, 3, 3)

        fx = K[:, 0, 0].unsqueeze(1).unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1).unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1).unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1).unsqueeze(1)

        x_rows = torch.arange(start=0, end=image_width, device=self.device)
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((image_height, 1))
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((batch_size, 1, 1))
        x_rows = x_rows.float()

        y_cols = torch.arange(start=0, end=image_height, device=self.device)
        y_cols = y_cols.unsqueeze(1)
        y_cols = y_cols.repeat((1, image_width))
        y_cols = y_cols.unsqueeze(0)
        y_cols = y_cols.repeat((batch_size, 1, 1))
        y_cols = y_cols.float()

        # 0.5 is so points are projected through the center of pixels
        x_scale = (x_rows + 0.5 - cx) / fx#; print(x_scale[0,0,:])
        y_scale = (y_cols + 0.5 - cy) / fy#; print(y_scale[0,:,0]); stop
        ones = (
            torch.ones((batch_size, image_height, image_width), device=self.device)
            .unsqueeze(3)
            .float()
        )
        return x_scale, y_scale, ones

    def point_cloud(self, depth, depth_scaling=1.0):
        """
        Converts image pixels to 3D pointcloud in camera reference using depth values.

        Args:
            depth (torch.FloatTensor): (batch_size, height, width)

        Returns:
            xyz1 (torch.FloatTensor): (batch_size, height * width, 4)

        Operation:
            z = d / scaling
            x = z * (u-cx) / fx
            y = z * (v-cv) / fy
        """
        shape = depth.shape
        if (
            shape[0] == self.batch_size
            and shape[1] == self.fmh
            and shape[2] == self.fmw
        ):
            x_scale = self.x_scale
            y_scale = self.y_scale
            ones = self.ones
        else:
            x_scale, y_scale, ones = self.compute_scaling_params(
                shape[0], shape[1], shape[2]
            )
        z = depth / float(depth_scaling)
        x = z * x_scale
        y = z * y_scale

        xyz1 = torch.cat((x.unsqueeze(3), y.unsqueeze(3), z.unsqueeze(3), ones), dim=3)
        return xyz1

    def transform_camera_to_world(self, xyz1, T):
        """
        Converts pointcloud from camera to world reference.

        Args:
            xyz1 (torch.FloatTensor): [(x,y,z,1)] array of N points in homogeneous coordinates
            T (torch.FloatTensor): camera-to-world transformation matrix
                                        (inverse of extrinsic matrix)

        Returns:
            (float tensor BxNx4): array of pointcloud in homogeneous coordinates

        Shape:
            Input:
                xyz1: (batch_size, 4, no_of_points)
                T: (batch_size, 4, 4)
            Output:
                (batch_size, 4, no_of_points)

        Operation: T' * R' * xyz
                   Here, T' and R' are the translation and rotation matrices.
                   And T = [R' T'] is the combined matrix provided in the function as input
                           [0  1 ]
        """
        return torch.bmm(T, xyz1)

    def pixel_to_world_mapping(self, depth_img_array, T):
        """
        Computes mapping from image pixels to 3D world (x,y,z)

        Args:
            depth_img_array (torch.FloatTensor): Depth values tensor
            T (torch.FloatTensor): camera-to-world transformation matrix (inverse of
                                        extrinsic matrix)

        Returns:
            pixel_to_world (torch.FloatTensor) : Mapping of one image pixel (i,j) in 3D world
                                                      (x,y,z)
                    array cell (i,j) : (x,y,z)
                        i,j - image pixel indices
                        x,y,z - world coordinates

        Shape:
            Input:
                depth_img_array: (N, height, width)
                T: (N, 4, 4)
            Output:
                pixel_to_world: (N, height, width, 3)
        """

        # Transformed from image coordinate system to camera coordinate system, i.e origin is
        # Camera location  # GEO:
        # shape: xyz1 (batch_size, height, width, 4)
        xyz1 = self.point_cloud(depth_img_array)

        # shape: (batch_size, height * width, 4)
        xyz1 = torch.reshape(xyz1, (xyz1.shape[0], xyz1.shape[1] * xyz1.shape[2], 4))

        # shape: (batch_size, 4, height * width)
        xyz1_t = torch.transpose(xyz1, 1, 2)  # [B,4,HxW]

        # Transformed points from camera coordinate system to world coordinate system  # GEO:
        # shape: xyz1_w(batch_size, 4, height * width)
        xyz1_w = self.transform_camera_to_world(xyz1_t, T)

        # shape: (batch_size, height * width, 3)
        world_xyz = xyz1_w.transpose(1, 2)[:, :, :3]

        # -- shift world origin
        world_xyz -= self.world_shift_origin

        # shape: (batch_size, height, width, 3)
        pixel_to_world = torch.reshape(world_xyz,((depth_img_array.shape[0],depth_img_array.shape[1],depth_img_array.shape[2],3,)),)

        return pixel_to_world

    def discretize_point_cloud(self, point_cloud, camera_height):
        """ #GEO:
        Maps pixel in world coordinates to an (output_height, output_width) map.
        - Discretizes the (x,y) coordinates of the features to gridcellsize.
        - Remove features that lie outside the (output_height, output_width) size.
        - Computes the min_xy and size_xy, and change (x,y) coordinates setting min_xy as origin.

        Args:
            point_cloud (torch.FloatTensor): (x,y,z) coordinates of features in 3D world
            camera_height (torch.FloatTensor): y coordinate of the camera used for deciding
                                                      after how much height to crop

        Returns:
            pixels_in_map (torch.LongTensor): World (x,y) coordinates of features discretized
                                    in gridcellsize and cropped to (output_height, output_width).

        Shape:
            Input:
                point_cloud: (batch_size, features_height, features_width, 3)
                camera_height: (batch_size)
            Output:
                pixels_in_map: (batch_size, features_height, features_width, 2)
        """
        
        # -- /!\/!\
        # -- /!\ in Habitat-MP3D y-axis is up. /!\/!\
        # -- /!\/!\
        pixels_in_map = ((point_cloud[:, :, :, [0,2]])/ self.gridcellsize).round()
        
        # Anything outside map boundary gets mapped to (0,0) with an empty feature
        # mask for outside map indices
        outside_map_indices = (pixels_in_map[:, :, :, 0] >= self.output_width) +\
                              (pixels_in_map[:, :, :, 1] >= self.output_height) +\
                              (pixels_in_map[:, :, :, 0] < 0) +\
                              (pixels_in_map[:, :, :, 1] < 0)
        
        # shape: camera_y (batch_size, features_height, features_width)
        camera_y = (camera_height.unsqueeze(1).unsqueeze(1).repeat(1, pixels_in_map.shape[1], pixels_in_map.shape[2]))
        
        # Anything above camera_y + z_clip_threshold will be ignored
        above_threshold_z_indices = point_cloud[:, :, :, 1] > (camera_y + self.z_clip_threshold)
        
        mask_outliers = outside_map_indices + above_threshold_z_indices

        return pixels_in_map.long(), mask_outliers


class PointCloud(ProjectorUtils):
    """
    Unprojects 2D depth pixels in 3D
    """

    def __init__(
        self,
        vfov,
        batch_size,
        feature_map_height,
        feature_map_width,
        map_dim,
        map_res,
        world_shift_origin,
        z_clip_threshold,
        device=torch.device("cuda"),
    ):
        """Init function

        Args:
            vfov (float): Vertical Field of View
            batch_size (float)
            feature_map_height (int): height of image
            feature_map_width (int): width of image
            world_shift_origin (float, float, float): (x, y, z) shift apply to position the map in the world coordinate system.
            z_clip_threshold (float): in meters. Pixels above camera height + z_clip_threshold will be ignored. (mainly ceiling pixels)
            device (torch.device, optional): Defaults to torch.device('cuda').
        """

        ProjectorUtils.__init__(self,
                                vfov,
                                batch_size,
                                feature_map_height,
                                feature_map_width,
                                1,
                                1,
                                1,
                                world_shift_origin,
                                z_clip_threshold,
                                device)

        self.vfov = vfov
        self.batch_size = batch_size
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.map_dim = map_dim
        self.map_res = map_res
        self.world_shift_origin = world_shift_origin
        self.z_clip_threshold = z_clip_threshold
        self.device = device


    def forward(self, depth, T, obs_per_map=1):
        """Forward Function

        Args:
            depth (torch.FloatTensor): Depth image
            T (torch.FloatTensor): camera-to-world transformation matrix
                                        (inverse of extrinsic matrix)
            obs_per_map (int): obs_per_map images are projected to the same map

        Returns:
            mask (torch.FloatTensor): mask of outliers. Mainly when no depth is present.
            point cloud (torch.FloatTensor)

        """

        assert depth.shape[2] == self.fmh
        assert depth.shape[3] == self.fmw

        depth = depth[:,0,:,:]

        # -- filter out the semantic classes with depth == 0. Those sem_classes map to the agent
        # itself .. and thus are considered outliers
        no_depth_mask = depth == 0

        # Feature mappings in the world coordinate system where origin is somewhere but not camera
        # # GEO:
        # shape: features_to_world (N, features_height, features_width, 3)
        point_cloud = self.pixel_to_world_mapping(depth, T)

        return point_cloud, no_depth_mask
    
    
    @torch.no_grad()
    def project_bev(self, pc, no_depth_mask, pc_feat):
        '''
        pc: (bs, N, 3)
        no_depth_mask: (bs, N)
        pc_feat: (bs, N, 768)
        '''
        bevs, ob_masks = [], []

        for pc_i, no_depth_mask_i, pc_feat_i in zip(pc, no_depth_mask, pc_feat):
            # discretize point cloud
            vertex_to_rgb_ft = pc_feat_i.clone()
            vertex_to_map_xz = (pc_i[:, [0,2]]/self.map_res + (self.map_dim-1)/2).round()
            outside_mask = (vertex_to_map_xz[:, 0] >= self.map_dim) +\
                           (vertex_to_map_xz[:, 1] >= self.map_dim) +\
                           (vertex_to_map_xz[:, 0] < 0) +\
                           (vertex_to_map_xz[:, 1] < 0)
            above_mask = (pc_i[:, 1] >  0.5)
            mask = no_depth_mask_i | outside_mask | above_mask
            vertex_to_rgb_ft = vertex_to_rgb_ft[~mask]
            vertex_to_map_xz = vertex_to_map_xz[~mask]

            # scatter feat
            feat_index = (self.map_dim * vertex_to_map_xz[:, 1] + vertex_to_map_xz[:, 0]).long()
            flat_bev = scatter_mean(vertex_to_rgb_ft, 
                                    feat_index, 
                                    dim=0, 
                                    dim_size=self.map_dim*self.map_dim)
            bev = flat_bev.reshape(self.map_dim, self.map_dim, 768)
            ob_mask = ~((bev.max(dim=-1)[0]==0) & (bev.min(dim=-1)[0]==0))
            bevs.append(bev)
            ob_masks.append(ob_mask)

        bevs = torch.stack(bevs, dim=0)
        ob_masks = torch.stack(ob_masks, dim=0)

        return bevs, ob_masks


def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        default="train",
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default="run_r2r/iter_train.yaml",
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument('--local_rank', type=int, default=0, help="local gpu id")
    args = parser.parse_args()
    config = run_exp(**vars(args))
    
    return config


def run_exp(exp_name: str, exp_config: str, 
            run_type: str, opts=None, local_rank=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()

    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.VIDEO_DIR += exp_name
    # config.TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.max_text_len = config.IL.max_text_len
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE

    if 'CMA' in config.MODEL.policy_name and 'r2r' in config.BASE_TASK_CONFIG_PATH:
        config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}.json.gz'

    config.local_rank = local_rank
    config.freeze()
    
    return config


def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done
    :return: VectorEnv object created according to specification.
    """

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    logger.info(f"SPLTI: {config.TASK_CONFIG.DATASET.SPLIT}, NUMBER OF SCENES: {len(scenes)}")

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )

        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)
    # import pdb;pdb.set_trace()
    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

            proc_config.freeze()
            configs.append(proc_config) 
    # import pdb;pdb.set_trace()
    is_debug = True if sys.gettrace() else False
    env_entry = habitat.ThreadedVectorEnv if is_debug else habitat.VectorEnv
    envs = env_entry(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)), 
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs


def transfrom3D(xyzhe):
    '''
    Return (N, 4, 4) transformation matrices from (N,5) x,y,z,heading,elevation 
    '''
    theta_x = xyzhe[:,4] # elevation
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)

    theta_y = xyzhe[:,3] # heading
    cy = np.cos(theta_y)
    sy = np.sin(theta_y)

    T = np.zeros([xyzhe.shape[0], 4, 4])
    T[:,0,0] =  cy
    T[:,0,1] =  sx*sy
    T[:,0,2] =  cx*sy 
    T[:,0,3] =  xyzhe[:,0] # x

    T[:,1,0] =  0
    T[:,1,1] =  cx
    T[:,1,2] =  -sx
    T[:,1,3] =  xyzhe[:,1] # y

    T[:,2,0] =  -sy
    T[:,2,1] =  cy*sx
    T[:,2,2] =  cy*cx
    T[:,2,3] =  xyzhe[:,2] # z

    T[:,3,3] =  1
    return T.astype(np.float32)


def heading_from_quaternion(quat: np.array):
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    quat = quaternion_from_coeff(quat)
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)


def bevpos_polar(map_dim):
    ''' bev position encoding '''
    ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, map_dim - 0.5, map_dim, dtype=torch.float32),
                                    torch.linspace(0.5, map_dim - 0.5, map_dim, dtype=torch.float32))
    ref = torch.stack([ref_y, ref_x], dim=-1)
    ref -= map_dim/2
    ref[..., 0] *= -1 # flip y axis
    map_dis = (ref[..., 0]**2 + ref[..., 1]**2) ** 0.5
    map_cos = ref[..., 1] / map_dis
    map_sin = ref[..., 0] / map_dis
    
    # for divide 0
    map_cos[map_dis==0] = 0
    map_sin[map_dis==0] = 0

    # normalize distance
    max_dis = map_dim/2
    map_dis /= max_dis
    map_pos = torch.stack([map_cos, map_sin, map_dis], dim=-1)
    return map_pos


def lift(envs, projector, cur_pos, cur_ori, rgb_grid, depth_grid):
    '''  unproject rgbs and depths to pointcloud in world coord '''
    bs = envs.num_envs
    num_view = 12

    xyzhe = np.zeros([bs, num_view, 5])
    for i, (pos_i, ori_i) in enumerate(zip(cur_pos, cur_ori)):
        x, y, z = pos_i
        xyzhe[i, :, 0] = x
        xyzhe[i, :, 1] = y
        xyzhe[i, :, 2] = z
        xyzhe[i, :, 3] = -np.arange(12) * np.radians(30) + heading_from_quaternion(ori_i)
        xyzhe[i, :, 4] = np.pi
    T = transfrom3D(xyzhe.reshape(-1,5))
    T = torch.from_numpy(T).cuda()  # bs, NUM_VIEW, 4, 4

    depths = depth_grid.reshape(-1, 1, 14, 14).cuda() * 10
    pc, pc_mask = projector.forward(depths, T)
    pc = pc.reshape(bs, -1, 3)
    pc_mask = pc_mask.reshape(bs, -1)
    
    rgbs = rgb_grid.reshape(-1, 14, 14, 768).cuda()
    pc_feat = rgbs.reshape(bs, -1, 768)

    return pc, pc_mask, pc_feat


def splat(envs, projector, cur_pos, cur_ori, pc, pc_mask, pc_feat):
        '''
            1. transform pointcloud to ego coord
            2. project to bev
        '''
        bs = envs.num_envs

        S = []
        for i, pos_i in enumerate(cur_pos):
            x, y, z = pos_i
            S.append([np.array([x, y, z])])
        S = np.vstack(S).astype(np.float32)              # bs, 3
        S = torch.from_numpy(S).cuda()
        xyzhe = np.zeros([bs, 5])
        for i, ori_i in enumerate(cur_ori):
            xyzhe[i, 3] = -heading_from_quaternion(ori_i)
        T = torch.from_numpy(transfrom3D(xyzhe)).cuda()  # bs, 4, 4

        # transform to ego coord
        pc = pc - S[:, None, :]
        ones = torch.ones(pc.shape[:2]).unsqueeze(-1).cuda()
        pc1 = torch.cat([pc, ones], dim=-1)               # bs, N, 4
        pc1 = torch.matmul(pc1, T.transpose(1, 2))        # bs, N, 4
        pc = pc1[:, :, :3]                                # bs, N, 3

        # project to bev
        # viz = False
        bev_fts, bev_masks = projector.project_bev(pc, pc_mask, pc_feat)
        # if viz:
        #     feat_masks = []
        #     for ob, bev_mask in zip(obs, bev_masks):
        #         cand_pos = self._map_cand_to_bev(ob)
        #         bev_mask = bev_mask.cpu().numpy()[:,:,None] * np.array([255,255,255])[None,None,:]
        #         bev_mask = bev_mask.astype(np.uint8)
        #         for p in cand_pos:
        #             bev_mask[p[1], p[0], :] = np.array([0,255,0]).astype(np.uint8)
        #         feat_masks.append(bev_mask)
        #     feat_masks = np.concatenate(feat_masks, axis=1)
        #     cv2.imwrite('feat_masks.png', feat_masks)

        #     bev_imgs = [draw_ob(ob) for ob in obs]
        #     bev_imgs = np.concatenate(bev_imgs, axis=0)
        #     cv2.imwrite('bev_imgs.png', bev_imgs)

        bev_fts = bev_fts.reshape(bs, -1, 768)
        bev_masks = bev_masks.reshape(bs, -1)
        
        bev_pos = bevpos_polar(bev_dim=11)
        bev_pos = bev_pos[None,:,:,:] # 1 x 11 x 11 x 3
        
        bev_pos_fts = bev_pos.expand(bs,-1,-1,-1).reshape(bs, -1, 3)

        return bev_fts, bev_masks, bev_pos_fts


if __name__ == "__main__":
    config = get_configs()
    envs = construct_envs(
            config, 
            get_env_class(config.ENV_NAME),
            episodes_allowed=None,
            auto_reset_done=False, # unseen: 11006 
        )
    projector = PointCloud(math.radians(90),
                    1,
                    feature_map_height=14,
                    feature_map_width=14,
                    map_dim=11,
                    map_res=1,
                    world_shift_origin=torch.FloatTensor([0,0,0]),
                    z_clip_threshold=0.5,
                    device=torch.device("cuda:4"))
    
    import pdb;pdb.set_trace()