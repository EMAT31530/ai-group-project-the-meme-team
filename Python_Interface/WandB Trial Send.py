#Before running software everyone neeeds to install and login from command line

# pip install --upgrade wandb
# wandb login 0c3e2969f4bf0251292bb09b1a6c6ed913a39d37

#Initialisation for wandb

import wandb

#set experiment name to group training and evaluation data

experiment_name = wandb.util.generate_id()

#Pointing to wandb project 

wandb.init(name='Rocket_Lander_2021.02.21_1202_JAH', 
           project='rocketlander',
           notes='This is a test run', 
           tags=['RocketLander', 'Test Run'],
           entity='uob_rocket_lander')

#This bit logs the metrics to W&B
for _ in range(max_steps):
    
       wandb.log({"Entropy":RocketLander.Policy.Entropy,
                  "Extrinsic Value Estimate": RocketLander.Policy.ExtrinsicValueEstimate, 
                  "Episode Length":RocketLander.Environment.EpisodeLength,
                  "Cumulative Reward":RocketLander.Environment.CumulativeReward, 
                  "Extrinsic Reward": RocketLander.Policy.ExtrinsicReward, 
                  "Policy Loss": RocketLander.Losses.PolicyLoss,
                  "Value Loss": RocketLander.Losses.ValueLoss,
                 })
# Insert training script here

#If you want to add summary analysis etc

wandb.log.run.summary["accuracy"]= best_accuracy

