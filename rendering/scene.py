import sapien
import torch
import numpy as np

from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import SO100
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.utils import randomization
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig

@register_env("PickPlaceCube-v1", max_episode_steps=50)
class PickPlaceCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["so100"]
    agent : SO100

    def __init__(
            self, init_robot_qpos_noise=0.02,
            init_cube_qpos_noise=0.02, init_xy_rad_noise=(-np.pi/4, np.pi/4),
            init_goal_qpos_noise=0.06, control_freq=20, render_freq=20,
            enable_shadow=True, robot_uids="so100",
            goal_radius=0.05, cube_half=0.02,
            *args, **kwargs
        ) -> None:

        self.init_robot_qpos_noise = init_robot_qpos_noise
        self.init_cube_qpos_noise = init_cube_qpos_noise
        self.init_xy_rad_noise = init_xy_rad_noise
        self.init_goal_qpos_noise = init_goal_qpos_noise
        self.goal_radius = goal_radius
        self.cube_half = cube_half
        self.device = torch.device("cuda")

        sim_cfg = SimConfig(
            control_freq=control_freq,
            sim_freq=render_freq,
            gpu_memory_config=GPUMemoryConfig(
            found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        ) 

        super().__init__(
            enable_shadow=enable_shadow, 
            robot_uids=robot_uids,
            sim_config=sim_cfg, 
            *args, **kwargs
        )

    @property
    def _default_sensor_configs(self):  
        cam_pose = sapien_utils.look_at(eye=[-0.9, 0.4, 0.3], target=self.cube.initial_pose.p, device=self.device)
        cam_pose = cam_pose * Pose.create_from_pq(
            p = torch.rand(3) * 0.05 - 0.025,
            q = randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi/24, np.pi/24)
            ),
            device=self.device
        )
        return [
            CameraConfig(
                "base_camera", pose=cam_pose, width=128, height=128, fov=np.pi/2
            )
        ]
    
    @property
    def _default_human_render_camera_configs(self):
        cam_pose = sapien_utils.look_at(eye=[-0.9, 0.4, 0.3], target=self.cube.initial_pose.p, device=self.device)
        return [
            CameraConfig(
                "render_camera", pose=cam_pose, width=128, height=128, fov=np.pi/2
            )
        ]
    
    @property
    def randomise_qpos_cube(self) -> Pose:
        with torch.device(self.device):
            cube_noise_pose = Pose.create_from_pq(
                p=torch.randn((self.num_envs), 3) * self.init_cube_qpos_noise,
                q=randomization.random_quaternions(
                    n=self.num_envs, 
                    bounds=self.init_xy_rad_noise, 
                    lock_x=True, lock_y=True
                ),
            )
            cube_noise_pose.p[..., 2] = 0 # no z-level translation
        return cube_noise_pose
    
    @property
    def randomise_qpos_goal(self) -> Pose:
        with torch.device(self.device):
            goal_noise_pose = Pose.create_from_pq(
                p=torch.randn((self.num_envs), 3) * self.init_cube_qpos_noise,
                q=randomization.random_quaternions(
                    n=self.num_envs, 
                    bounds=self.init_xy_rad_noise, 
                    lock_x=True, lock_y=True
                ),
            )
            goal_noise_pose.p[..., 2] = 0 # no z-level translation
        return goal_noise_pose
    
    def _load_lighting(self, options):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [
                np.random.uniform(0.4, 0.8), 
                np.random.uniform(0.4, 0.8), 
                np.random.uniform(0.4, 0.8)
            ]
        pass

    def _load_agent(self, options : dict) -> None :

        super()._load_agent(options, sapien.Pose(
            p=[0, 0, 1]
        ))
    
    def _load_scene(self, options : dict) -> None:

        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.init_robot_qpos_noise
        )
        self.table_scene.build()

        self.cube = actors.build_cube(
            scene=self.scene,
            half_size=self.cube_half,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.5, -0.3, self.cube_half])
        )
        self.goal_region = actors.build_box(
            scene=self.scene,
            half_sizes=[0.1, 0.1, 0.1],
            color=np.array([12, 42, 160, 255]) / 255,
            body_type="kinematic",
            name="goal",
            initial_pose=sapien.Pose(p=[-0.5, -0.3, 1e-3])
        )

    def _initialize_episode(self, env_idx, options) -> None:

        with torch.device(self.device):
            
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.cube.set_pose(
                self.cube.pose * self.randomise_qpos_cube
            )
            self.goal_region.set_pose(
                self.goal_region.pose * self.randomise_qpos_goal
            )

    def evaluate(self) -> dict:
        
        return {
            "success" : torch.linalg.norm(
                self.goal_region.pose.p[..., :2] - self.cube.pose.p[..., :2]
            ) < self.goal_radius
        }

    def compute_normalized_dense_reward(self, obs, action, info) -> float:

        return 0