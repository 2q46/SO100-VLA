import sapien
import torch
import numpy as np
import gymnasium as gym 
import pyroki as pk
import jax.numpy as jnp
import yourdfpy
from pathlib import Path
from solver import _solve_ik_jax_batched    
from transforms3d.euler import euler2quat
from agent import SO100Base
from mani_skill.agents.robots import SO100
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.controllers import PDJointPosController
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig

@register_env("PickCubeCustom-v1", max_episode_steps=200)
class PickCubeCustom(BaseEnv):

    SUPPORTED_ROBOTS = ["so100_base"]
    agent : SO100Base
    def __init__(
        self, init_robot_qpos_noise=0.02, 
        init_cube_p_noise=0.02, init_cube_angle_noise=np.pi/3,
        robot_uids="so100_base", half_cube_dim=0.02, goal_radius=0.03,
        init_goal_p_noise=0.02,
        *args, **kwargs
        ):
        self.device = torch.device("cuda")
        super().__init__(robot_uids=robot_uids, *args, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)
        ]
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.1, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100)
    
    def _load_agent(self, options: dict):
        return super()._load_agent(options)
    
    def _load_scene(self, options):
        self.ground = build_ground(self.scene)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[0.02]*3
        )
        builder.add_box_visual(
            half_size=[0.02]*3,
            material=sapien.render.RenderMaterial(
                base_color=[1, 0, 0, 1],
            ),
        )
        builder.initial_pose = sapien.Pose(
            p=[-0.17, -0.17, 0.02],
            q=[1, 0, 0, 0]
        )
        self.cube = builder.build(name="cube")
        return super()._load_scene(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        with torch.device(self.device):      

            batch = len(env_idx)
            self.cube.set_pose(self.cube.initial_pose)

    def evaluate(self):
        return {
            "success": torch.linalg.norm(
                torch.tensor(0.0, device=self.device)
            ) < 0
        }

    def compute_normalized_dense_reward(self, obs, action, info):
        return 0

env = gym.make(
    id="PickCubeCustom-v1",
    num_envs=1,
    render_mode="human",
    obs_mode="state_dict",
)

obs, _ = env.reset(seed=0)
urdf_path = Path("so100/so100.urdf")
urdf = yourdfpy.URDF.load(urdf_path)
robot = pk.Robot.from_urdf(urdf)

target_positions = [
    jnp.array([[-0.2, -0.2, 0.05]]),
    jnp.array([[-0.2, -0.2, 0.05]]),
] 

target_quaternions = [
    jnp.array([euler2quat(0, -np.pi/2, 0)]),
    jnp.array([euler2quat(0, -np.pi/2, 0)])
]

solutions = []

for position, quat in zip(target_positions, target_quaternions):
 
    solution = _solve_ik_jax_batched(robot, 6, quat, position)
    solution = torch.tensor(np.array(solution)).cuda()
    solutions.append(solution)

print(solutions)

'''
done = False
while True:
    
    for sol in solutions:
        obs, reward, terminated, truncated, info = env.step(sol)
        done = truncated or terminated
        env.render()
'''