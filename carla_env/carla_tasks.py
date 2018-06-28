__all__ = [
    "CarlaLaneFollowEnv",
    "CarlaLaneFollowCarEnv",
    "CarlaRacerEnv",
    "CarlaPassEnv"
]

import numpy as np
import gym

from sorl.envs.carla_env.carla_env import CarlaEnv

def run_carla_env(env):
    env = gym.wrappers.Monitor(env(), "./env_logs", force=True)
    try:
        env.reset()
        for i in range(1000):
            m, r, done, _ = env.step([0, 1, 0])
            if done:
                break
        for i in range(3):
            env.reset()
            for i in range(1000):
                a = np.random.uniform(-1,1) if i % 10 == 0 else a
                m, r, done, _ = env.step([a, np.random.uniform(0.5, 1), 0])
                print(r)
                if done:
                    break
    finally:
        env.unwrapped.close()

class CarlaLaneFollowEnv(CarlaEnv):
    def __init__(self, **kwargs):
        super().__init__(num_vehicles=0, **kwargs)

    def _get_reward_and_termination(self):
        reward, is_done = super()._get_reward_and_termination()

        measurements, _ = self.current_state

        off_lane = measurements.player_measurements.intersection_otherlane

        reward += (1 - off_lane) * (300 - self.dist_from_goal(measurements))
        is_done = is_done or off_lane > 0.7
        return reward, is_done

class CarlaLaneFollowCarEnv(CarlaLaneFollowEnv):
    def __init__(self, **kwargs):
        super().__init__(num_vehicles=1, **kwargs)

class CarlaRacerEnv(CarlaEnv):
    pass

class CarlaPassEnv(CarlaEnv):
    pass


if __name__ == "__main__":
    run_carla_env(CarlaLaneFollowEnv)