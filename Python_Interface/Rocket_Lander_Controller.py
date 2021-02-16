"""
Rocket_Lander_Controller.py
Description: This file acts as a python interface for Unity ML-Agent training
"""

from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from PPO_Implementation import PPO

"""
HYPERPARAMETERS===========================================================================================

steps: The number of steps to train the model on (how long to train), default = 1e8
learning_rate: The learning rate (the size of the update steps), default = 2.5e-4
num_epoch: The number of times to iterate through whole training dataset, default = 3
batch_size: The number of samples to be processed before model update, default = 1024
mini_batch_size: The number of samples to be processed within the batch per iteration,
            default = 256 (NOTE: mini_batch_size should be factor of batch_size)
                 
epsilon: PPO loss clipping parameter (defining threshold for objective improvement), default = 0.2
c1: Value function coefficient (scales contribution of value loss to combined loss), default = 1
c2: Entropy coefficient (scales contribution of entropy to combined loss), default = 0.01
hidden_1: The size of the first hidden layer, default = 32
hidden_2: The size of the second hidden layer, default = 32
custom_optimizer: What optimiser to use (refer to torch.optim), 
            default = None (which uses internally defined Adam)
    
gamma: Discount factor to prioritise immediate reward (in GAE), default = 0.99
lambd: Smoothing term for reducing variance in GAE, default = 0.95
gradient_clip_value: Clipping threshold for gradient norm to avoid exploding gradients, default = 0.5
custom_scheduler: What optimiser to use (refer to torch.optim.lr_scheduler),
            default = None (which uses internally defined StepLR)
    
scheduler_step: Frequency of scheduler LR update (if custom_scheduler=None), default = 20
scheduler_gamma: Scaling factor of scheduler LR update (if custom_scheduler=None), default = 0.9
state_dict: State dictionary of network weights, default = None
            Trained values can be loaded with torch.load("<FILENAME>")
    
checkpoint_freq: How often to write out / save state dictionary, default = 1e5
is_training: Whether the agent should is training, default = True
            Note: no state_dict should be passed if training, and vice versa

==========================================================================================================
"""

steps = 1e8
learning_rate = 2.5e-4
num_epoch = 3
batch_size = 1024
mini_batch_size = 256
epsilon = 0.2
c1 = 1
c2 = 0.01
hidden_1 = 32
hidden_2 = 32
custom_optimizer = None
gamma = 0.99
lambd = 0.95
gradient_clip_value = 0.5
scheduler_step = 20
scheduler_gamma = 0.9
custom_scheduler = None
state_dict = None
checkpoint_freq = 1e5

is_training = True

# Creates a tensorboard summary writer
# TODO - only use tensorboard for training
writer = SummaryWriter() 


"""
ENVIRONMENT SETUP==========================================================================================

is_headless: Set True to skip rendering environment, defalt = False
time_scale: The speed the environment will be run at (scale mulitple), default = 1.0
width: The width (px) of the displayed window if is_headless = False, default = 500
height: The height (px) of the displayed window if is_headless = False, default = 500

==========================================================================================================
"""

is_headless = False
time_scale = 1.0
width = 500
height = 500

# Setup pytorch with predefined seed for consistency / reproducability
torch.manual_seed(1)

# Create environment
channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name="Unity_Compiled_Files/Rocket_Lander", seed=1, side_channels=[channel], no_graphics = is_headless)
channel.set_configuration_parameters(time_scale = time_scale, width = width, height = height)
env.reset()


"""
MAIN EVENT LOOP===========================================================================================

Main logic to run the environment in train / test mode, and update

==========================================================================================================
"""

if not is_training:
    if bool(state_dict):
        ppo = PPO(env, state_dict = state_dict, learning_rate = learning_rate, num_epoch = num_epoch,
                 batch_size = batch_size, mini_batch_size = mini_batch_size, epsilon = epsilon, c1 = c1, 
                 c2 = c2, hidden_1 = hidden_1, hidden_2 = hidden_2, custom_optimizer = custom_optimizer,
                 gamma = gamma, lambd = lambd, gradient_clip_value = gradient_clip_value,
                 scheduler_step = scheduler_step, scheduler_gamma = scheduler_gamma,
                 writer = writer, custom_scheduler = custom_scheduler)
    else:
        env.close()
        raise("No state dictionary (of weights) provided to run neural network!")
else:

    # Configure to ignore state_dict if provided
    ppo = PPO(env, state_dict = None, learning_rate = learning_rate, num_epoch = num_epoch,
                 batch_size = batch_size, mini_batch_size = mini_batch_size, epsilon = epsilon, c1 = c1, 
                 c2 = c2, hidden_1 = hidden_1, hidden_2 = hidden_2, custom_optimizer = custom_optimizer,
                 gamma = gamma, lambd = lambd, gradient_clip_value = gradient_clip_value,
                 scheduler_step = scheduler_step, scheduler_gamma = scheduler_gamma,
                 writer = writer, custom_scheduler = custom_scheduler)

    # Extract current datestamp, and use to prefix files
    datestamp = datetime.now().strftime("%Y.%m.%d_%H%M")
    checkpoint_num = 1


# TODO - Check correct tensorboard / output logging
# TODO - Decide where to save out state dictionary, etc.
iterations = steps / batch_size
for iteration in range(int(iterations)):

    try:
        print("Iteration No.", iteration)
        total_observations, total_values, total_rewards, total_actions, masks, advantage, values = ppo.policy_rollout(batch_size)

        if is_training:

            # Write value and policy loss data to tensorboard
            # TODO - account for correct steps
            value_loss, policy_loss = ppo.update(None, total_observations, total_actions, advantage, values)
            writer.add_scalar("Losses/Value_Loss", value_loss, iteration * batch_size)
            writer.add_scalar("Losses/Policy_Loss", policy_loss, iteration * batch_size)

            # TODO = further tensorboard placeholders
            """
            writer.add_scalar("Environment/Cumulative_Reward", , )
            writer.add_scalar("Environment/Episode_Length", , )

            writer.add_scalar("Policy/Entropy", , )
            writer.add_scalar("Policy/Extrinsic_Reward", , )
            writer.add_scalar("Policy/Extrinsic_Value_Estimate", , )
            """

            # Save out checkpoint state dictionary if checkpoint frequency exceeded
            if (iteration * batch_size) /  checkpoint_freq > checkpoint_num:
                torch.save(ppo.net.state_dict(), datestamp + "_RocketLander_Checkpoint_" + str(checkpoint_num))
                print("SAVING: ", datestamp + "_RocketLander_Checkpoint_" + str(checkpoint_num))                
                checkpoint_num += 1

                
    #Exception handling for escaping environment with Escape key
    except Exception as e:
        print(e)
        break

# Save out final state dictionary when training stopped
if is_training:
    torch.save(ppo.net.state_dict(), datestamp + "_RocketLander_Final")
    print("SAVING: ", datestamp + "_RocketLander_Final")

# Close the environment and tensorboard writer when finished
writer.flush()
writer.close()
env.close()

