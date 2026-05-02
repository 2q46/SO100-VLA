import gymnasium as gym
from scene import PickPlaceCubeEnv
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig

env = gym.make(
    "PickPlaceCube-v1",
    num_envs=4,
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="human", # human, sensors
)
'''
env = RecordEpisode(
    env=env,
    output_dir="videos/",
    save_trajectory=False,
    max_steps_per_video=120
) 
'''
print(f"Observation space: {env.observation_space}")
print(f"Observation space: {env.action_space}")

obs, _ = env.reset(seed=0)
done = False
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
env.close()