import torch
import pyroki as pk
import yourdfpy
import numpy as np
from pathlib import Path
import gymnasium as gym
from solver import solve_ik
from scene import PickPlaceCubeEnv
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig

urdf_path = Path("SO100/so100.urdf")
urdf = yourdfpy.URDF.load(urdf_path)
target_link_name = "gripper"
robot = pk.Robot.from_urdf(urdf)

action = torch.zeros((4, 6), device=torch.device("cuda"))
positions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])
for i in range(4):
    delta = solve_ik(
        robot=robot,
        target_link_name=target_link_name,
        target_position=positions[i],
        target_wxyz=np.array([1, 0, 0, 0])
    )
    action[i] = torch.tensor(delta, device=torch.device("cuda"))

env = gym.make(
    "PickPlaceCube-v1",
    num_envs=4,
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="sensors", # human, sensors
)

env = RecordEpisode(
    env=env,
    output_dir="videos/",
    save_trajectory=False,
    max_steps_per_video=120
) 

print(f"Observation space: {env.observation_space}")
print(f"Observation space: {env.action_space}")
print(env.action_space.sample())

obs, _ = env.reset(seed=0)
done = False
while True:
    env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
env.close()