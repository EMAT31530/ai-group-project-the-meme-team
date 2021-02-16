import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.base_env import ActionTuple


# Define the actor critic network - for generating a policy (action probabilities) and value of policy
class ActorCriticNetwork(nn.Module):

    def __init__(self, input_size = 16, hidden_1 = 32, hidden_2 = 32, action_size = 3):
        super(ActorCriticNetwork, self).__init__()

        # Policy network (actor)
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.Tanh(),
            nn.Linear(hidden_1, hidden_2),
            nn.Tanh(),
            nn.Linear(hidden_2, action_size),
            nn.Tanh()
            )

        # Value network (critic)
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.Tanh(),
            nn.Linear(hidden_1, hidden_2),
            nn.Tanh(),
            nn.Linear(hidden_2, 1)
            )


    def forward(self, x):

        # Compute actiona probabilities (mapped to range [-1, 1] by Tanh)
        action_probs = self.actor(x)

        # Compute value of policy
        value = self.critic(x)
        
        return action_probs, value


# Implementation of Proximal Policy Optimisation (PPO)
# As defined in https://arxiv.org/pdf/1707.06347.pdf
class PPO():
    def __init__(self, env, state_dict = None, learning_rate = 2.5e-4, num_epoch = 3,
                 batch_size = 1024, mini_batch_size = 256, epsilon = 0.2, c1 = 1, 
                 c2 = 0.01, hidden_1 = 32, hidden_2 = 32, custom_optimizer = None,
                 gamma = 0.99, lambd = 0.95, gradient_clip_value = 0.5,
                 scheduler_step = 20, scheduler_gamma = 0.9, custom_scheduler = None,
                 writer = None):

        # Store reference to the tensorboard summary writer
        self.writer = writer

        # Set environment and model parameters
        self.env = env
        self.behaviour_name = list(self.env.behavior_specs.keys())[0]
        self._num_agents = len(list(self.env._env_state[self.behaviour_name][0]))
        self.input_size = list(env.behavior_specs[self.behaviour_name])[0][0][0]
        self.action_size = list(env.behavior_specs[self.behaviour_name])[1][0]

        # Create a covariance matrix for actions
        # Populate a compatible matrix with action_std, then convert to diagonal matrix
        self.action_std = 0.5
        self.action_variance = torch.full(size=(self.action_size,), fill_value = self.action_std)
        self.covariance_matrix = torch.diag(self.action_variance)

        # Setting Hyperparameters
        self.epochs = num_epoch # No of times to iterate through whole training dataset
        self.batch_size = batch_size # Number of samples to be processed before model update
        self.mini_batch_size = mini_batch_size # Number of samples to be processed per iteration
                                               # Note: should be factor of batch_size * num_agents
        
        self.gamma = gamma # Discount factor to prioritise immediate reward
        self.lambd = lambd # Smoothing term for reducing variance in GAE
        self.epsilon = epsilon # Loss clipping parameter (defining threshold for objective improvement)
        
        self.c1 = c1 # Value function coefficient (scales contribution of value loss)
        self.c2 = c2 # Entropy coefficient (scales contribution of entropy)
        self.gradient_clip_value = gradient_clip_value # Gradient clip parameter (max gradient norm)
        
        # Create actor critic network(s) - new and old (for updating between 2 policies)
        self.net = ActorCriticNetwork(self.input_size, hidden_1, hidden_2, self.action_size)
        self.old_net = ActorCriticNetwork(self.input_size, hidden_1, hidden_2, self.action_size)

        # Align the starting state_dict of new and old actor critic networks
        self.state_dict = self.net.state_dict()
        self.old_net.load_state_dict(self.state_dict)

        # If supplied a state dictionary, load this into the new network
        # Note: This should also lock / prevent training
        if bool(state_dict):
            self.net.load_state_dict(state_dict)
        else:
            self.state_dict = self.net.state_dict()

        # Assign optimiser, default: Adam
        if bool(custom_optimizer):
            self.optimizer = custom_optimizer
        else:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr = learning_rate)
            
        # Assign learning rate scheduler, default: step learning rate (decays by gamma every n steps)
        if bool(custom_scheduler):
            self.scheduler = custom_scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                            step_size = scheduler_step, gamma = scheduler_gamma)


    # Function to generate a distribution from action probabilities, and return information from this
    # Uses a multivariate normal (Gaussian) distribution centred about action probability (as continuous)
    def sample_actions(self, action_mean):
        
        # Create multivariate normal distribution from action means and variances, and sample it
        mvn = MultivariateNormal(action_mean, self.covariance_matrix)

        # Sample a random action from the multivariate normal distribution
        action = mvn.sample()

        # Calculate the log probability of taking the selected action from the distribution
        log_prob = mvn.log_prob(action)

        # Calculate the information entropy of the distribution (how random / unexpected)
        entropy = mvn.entropy()

        return action, log_prob, entropy
    

    # Function to calculate the Generalised Advantage Estimate (GAE)
    # As defined in https://ewrl.files.wordpress.com/2015/02/ewrl12_2015_submission_18.pdf
    def compute_GAE(self, rewards, values, masks):

        # Fetch dimensions of rewards array - for timesteps (T) and no. of agents (W)
        T, width = rewards.shape

        # Initialise arrays with zeros
        _values = np.zeros((T, width))
        advantages = np.zeros((T, width))
        advantage_t = np.zeros((width, 1)).squeeze()
        
        # Iterating through the timesteps in reverse (as discounted sum in time)
        for t in reversed(range(T)):
            # This is the Bellman error term at step t
            delta_t = rewards[t] + self.gamma * values[t + 1] * masks[t] - values[t]

            # This is the advantage estimator at step t
            advantage_t = delta_t + self.gamma * self.lambd * masks[t] * advantage_t
            advantages[t] = advantage_t

        # Accumulate values using estimated advantage
        _values = values[:T] + advantages
        return advantages, _values

    
    # Procedure to log values to tensorboard during policy rollout
    # TODO - Write implementation
    def tensorboard_log_values(self):
        pass
            

    # Rollout policy and find advantage estimates
    def policy_rollout(self, steps):

        # Initialise zero arrays for storing values over batch
        total_observations = np.zeros((steps, self._num_agents, self.input_size))
        total_rewards = np.zeros((steps, self._num_agents))
        total_actions = np.zeros((steps, self._num_agents, self.action_size))
        total_values = np.zeros((steps+1, self._num_agents))
        masks = np.ones((steps, self._num_agents))

        # Run (old) policy in environment for T timesteps
        for step in range(steps):
            # Fetch observation at time step, and add to accumulator array
            observations = np.array(self.env._env_state[self.behaviour_name][0].obs)
            
            total_observations[step] = observations

            # Forward pass of actor critic network, and store values for given step
            action_probs, value = self.net(torch.from_numpy(observations).type(torch.FloatTensor))
            total_values[step] = value.view(-1).detach().numpy()


            # Take action as sampled from the fitted probability distribution, and advance environment
            action, log_prob, entropy = self.sample_actions(action_probs)
            self.env.set_actions(self.behaviour_name, ActionTuple(action.squeeze().numpy()))
            self.env.step()
            
            # Populate masks by assigning those who have completed environment with 0
            masks[step][self.env._env_state[self.behaviour_name][1].agent_id] = 0

            # Store rewards at current step
            total_rewards[step] = self.env._env_state[self.behaviour_name][0].reward

            # Store executed actions at current step
            total_actions[step] = action.squeeze().numpy()

            """
            # TODO - Call logging function to write out value to tensorboard
            self.tensorboard_log_values()
            """

        # Compute advantge estimates (using GAE)
        advantages, values = self.compute_GAE(total_rewards, total_values, masks)

        # Normalise advantages for accelerated learning / convergence
        advantage = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Return all experience arrays from the policy rollout
        return total_observations, total_values, total_rewards, total_actions, masks, advantage, values

        
    # Optimise / update loss w.r.t theta, K epochs, and minibatch size M < NT
    def update(self, steps, total_observations, total_actions, advantage, values):

        steps = int(self.batch_size * self._num_agents)

        # Fetch and reshape array of actions
        total_actions = total_actions.reshape((steps, -1))

        # Read and convert arrays to torch tensors for use with the network
        total_observations_tensor = torch.from_numpy(total_observations).view(steps, -1, 1).type(torch.FloatTensor)
        advantage_tensor = torch.from_numpy(np.float64(advantage)).view(steps, -1).type(torch.FloatTensor)
        values_tensor = torch.from_numpy(np.float64(values)).view(steps, -1).type(torch.FloatTensor)

        # Iterate through epoch
        for k in range(self.epochs):

            # Generate random permutation of indices for mini-batch
            shuffled_indices = torch.randperm(self._num_agents * self.batch_size)

            # Iterate through different mini-batches
            for i in range(0, self._num_agents * self.batch_size, self.mini_batch_size):

                # Select subset of shuffled indices
                indices = shuffled_indices[i:i+self.mini_batch_size]
        
                # Find action probabilities from mini-batch of observations per policy
                action_probs, value_backprop = self.net(total_observations_tensor[indices].squeeze())
                old_probs, _ = self.old_net(total_observations_tensor[indices].squeeze())

                # Create distribution based on policy, and evaluate
                action, log_prob, entropy = self.sample_actions(action_probs)
                old_action, old_log_prob, old_entropy = self.sample_actions(old_probs.detach())

                # Find the probability ratio, r_t
                # r_t is given by the division of the action probability of the old policy by the new policy
                # Here, it is expressed here as the exponent of the difference of log probabilities
                ratio = torch.exp(log_prob - old_log_prob.detach())

                # Find conservative policy iteration term, L^(CPI)
                surrogate_1 = ratio * advantage_tensor[indices].squeeze()

                # Find modified (clipped) surrogate objective term, L^(CLIP)
                surrogate_2 = (torch.clamp(ratio, min=(1.0 - self.epsilon), max=(1.0 + self.epsilon))
                              * advantage_tensor[indices].squeeze())

                # Assign minimum objective as pessimistic (unclipped lower) bound for policy loss
                policy_loss = - torch.min(surrogate_1, surrogate_2)

                # Find value loss, L^VF, as squared error loss between predicted and actioned values
                value_loss = ((value_backprop.detach() - values.reshape((-1,1))[indices])**2)

                # Combine objectives for combined loss, L^(CLIP+VF+S), which is to be maximised
                # (Linear combination of policy loss, (scaled) value loss, and (scaled) entropy bonus)
                combined_loss = policy_loss.squeeze() + self.c1*value_loss.squeeze() - self.c2*entropy

                # Reset the optimiser gradient(s)
                self.optimizer.zero_grad()

                # Backpropagate the (mean) combined loss
                combined_loss.mean().backward()

                # Clip gradient norm to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clip_value)

                # Perform parameter update with clipped gradients
                self.optimizer.step()

            # Advance / update the learning rate scheduler
            self.scheduler.step()

            # Update the state dictionaries
            self.old_net.load_state_dict(self.state_dict)
            self.state_dict = self.net.state_dict()

            # Return mean value loss and policy loss
            return value_loss.mean(), policy_loss.mean()
