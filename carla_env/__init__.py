from carla_env.carla_tasks import *

from gym.envs.registration import register

register(
    id='CarlaLaneFollow-v0',
    entry_point='carla_env:CarlaLaneFollowEnv',
    max_episode_steps=1000
)

register(
    id='CarlaLaneFollowCar-v0',
    entry_point='carla_env:CarlaLaneFollowCarEnv',
    max_episode_steps=1000
)

register(
    id='CarlaPass-v0',
    entry_point='carla_env:CarlaPassEnv',
    max_episode_steps=1000
)

register(
    id='CarlaRacer-v0',
    entry_point='carla_env:CarlaRacerEnv',
    max_episode_steps=1000
)