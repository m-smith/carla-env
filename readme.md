


# An OpenAI gym wrapper for simple custom CARLA tasks

In order to perform RL research in the CARLA simulator with code that abstracts over environments, we
implement a self-contained set of CARLA tasks which implement the OpenAI gym environment API.

Please note that these tasks are still fairly simple and under development.

## Installation

```carla-env``` is easy to install. Simply clone the repo, enter it with `cd carla-env` and run `pip install -e .`  to install the package.

In your python code, the environment can then be imported with
```python
    import carla_env
```

Please note that the first time the environment is imported, the custom CARLA simulator packaged with this repo will be downloaded and extracted to the location of the `carla-env` folder. This may take some time.

## Implemented Environments

Currently, the list of environments that are implemented is:

* __`CarlaLaneFollow-v0`__: This environment is a simple setup in which a vehicle begins at the start of a straigtaway and must simply follow the lane until the end of the path. Rewards are proportional to how close the agent is to the goal, and penalties are given for exiting the lane, going offroad, or crashing into obstacles.


## Planned/Partially Implemented Environments

Some upcoming and in progress environments include:

* __`CarlaLaneFollowCar-v0`__: In the same straightaway as `CarlaLaneFollow-v0`, the agent must drive forwards as quickly as possible, but now a car spawns in front, preventing the agent from being able to simply accelerate.
* __`CarlaPass-v0`__: In the same setup as `CarlaLaneFollowCar-v0`, the agent now is able to leave its lane in order to pass the car in front, and must do so before the straightaway ends.

## Custom Environments

In order to create custom environments, one may subclass `carla_env.CarlaEnv`. Particularly relevant functions to override include:

* __`_add_settings(self)`__: this function must call `self.settings.set(**kwargs)` in order to customize the environment features and dynamics as available in the CARLA settings: [here](https://github.com/carla-simulator/carla/blob/64b1b27315b6554e9cdab53046e6014ba3e7536a/Docs/Example.CarlaSettings.ini)
* __`_add_sensors(self)`__: this function creates one or several carla sensor or camera object (as described [here](https://carla.readthedocs.io/en/stable/cameras_and_sensors/)), and adds it to the vehicle by calling `self.settings.add_sensor(camera)`
* __`_map_controls(self, a)`__: This function accepts an action taken by the agent and returns a `dict` representing the actual input to the carla environment (as sent to `client.send_control()` - see [here](https://github.com/carla-simulator/carla/blob/64b1b27315b6554e9cdab53046e6014ba3e7536a/PythonClient/client_example.py#L138) for an example. )
* __`_process_observation(self, measurements, sensor_data)`__: this function accepts the privelleged measurements given by CARLA, as well as the sensor data (according to the sensors added in `_add_sensors`), and returns a single `numpy` array representing the current environment state. This function can be used to provide the agent with privileged information (true position, velocity, distance to objects, etc.), or restrict it to only actual sensor data (camera image, lidar scan, etc.).
* __`_get_reward_and_termination(self)`__: this function uses information from the `(measurements, sensor_data)` tuple stored in `self.current_state` to compute the environment reward for the current time step, as well as detect if we have reached a terminal state. It returns a tuple `reward, is_done`

Also of note are the initialization parameters:

* __`num_vehicles`__: The number of vehicles to spawn
* __`vehicles_seed`__: A function returning the random seed for NPC vehicles spawn location
* __`player_starts`__: The player start location index. These can be checked by running the script `{path_to_carla-env}/carla_dist/PythonClient/view_start_positions.py`, while the CARLA client is running in server mode according to the instructions [here](https://carla.readthedocs.io/en/stable/connecting_the_client/). If given as a list, spawns will be selected randomly from the list each episode.
* __`goals`__: Same as player_starts, but for goal locations.

## Implementation Details

The environment spawns a carla server in a seperate thread using `Popen`. When the process calling the environment is done with it, it must call `env.close()` in order to clean up the threads created by this. I'm still not sure this is working 100% reliably, so be aware that upon repeated environment reinstantiation, some threads may be left over and need to be killed manually.
