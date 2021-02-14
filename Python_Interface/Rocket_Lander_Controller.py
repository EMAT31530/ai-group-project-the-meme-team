"""
Rocket_Lander_Controller.py
Description: This file acts as a python interface for Unity ML-Agent training
"""

# Import dependencies
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

# Set model hyperparameters - TO DO
num_episodes = 400

# Set environment parameters
is_headless = False
time_scale = 1.0
width = 500
height = 500


# Create environment
channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name="Rocket_Lander", seed=1, side_channels=[channel], no_graphics = is_headless)
channel.set_configuration_parameters(time_scale = time_scale, width = width, height = height)

env.reset()

# Extract the behaviour name from the environment
behaviour_name = list(list(env.behavior_specs.__dict__.values())[0].keys())[0]

# Temporary hardcoded actions
from mlagents_envs.base_env import ActionTuple
for episodes in range(num_episodes):

    try:
        # Generate actions (continuous, 3) for each agent (4)
        action = ActionTuple(np.array([[1.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0]], dtype=np.float32))

        # Set the agent actions
        env.set_actions(behaviour_name, action)

        # Step forward the environment (to the next decision step)
        env.step()

    #Exception handling for escaping environment with Escape key
    except Exception as e:
        break

# Close the environment when finished
env.close()
