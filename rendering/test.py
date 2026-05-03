import torch
import pyroki as pk
import yourdfpy
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import gymnasium as gym
from solver import _solve_ik_jax_batched
from scene import PickPlaceCubeEnv
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig

urdf_path = Path("SO100/so100.urdf")
urdf = yourdfpy.URDF.load(urdf_path)
target_link_name = "gripper"
robot = pk.Robot.from_urdf(urdf)

positions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])
solution = _solve_ik_jax_batched(
    robot=robot,
    target_link_index=jnp.array(5),
    target_position=jnp.array(positions),
    target_wxyz=jnp.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
)
action = torch.from_numpy(np.array(solution)).cuda()

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