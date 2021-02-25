#Wandb sign in and set up. Run as separate function/ cell in code.
import wandb
wandb.login

#Multiple types of search are available can be defined as below:
#Search for grid search change method to 'grid'. Bayesian is a little more advanced as commented out below
sweep_config = {
                'method': 'random'
                'parameters':{
                    #Hyperparameters taken from config.yaml file
                    'batch_size':{ 
                        'values': [2048] }
                    'buffer_size':{
                        'values': [5096] }
                    'learning_rate':{
                        'values': [0.00025]}
                    'beta':{
                        'values': [0.001]}
                    'epsilon':{
                        'values': [0.2]}
                    'lambd': {
                        'values': {[0.99]}
                    'num_epoch':{ 
                        'values': [3]}
                    learning_rate_schedule: linear
                    #network_settings:
                    normalize: true
                    'hidden_units':{
                        'values': [256]}
                    'num_layers': {
                        'values': [2]}
                    #reward_signals:
                    extrinsic:
                    'gamma':{
                        'values': [0.99]}
                    'strength':{
                        'values': [1.0]}
                    #'max_steps': 100000000.0
                    #'time_horizon': 64
                    #'summary_freq': 50000
                    }   
       
                }
                
# #Bayesian Search method is titled 'bayes'
# Also requires additional metric definition:
# metric = {
    # 'name': 'loss',
    # 'goal': 'minimize'   #if want sle. for accuracy, 'minimize' should be the 'goal'
    # }

#sweep_config['metric'] = metric


#To intialise the parameter sweep: this should provide a link to the sweep in the browser to use and track the runs.
sweep_id= wandb.sweep(sweep_config)

# The agent then needs to run as the corresponding function naming converntion below:

def train():

#initialise wandb for run - V IMPORTANT
wandb.init(project = "rocketlander")

#execute the function to train the agent and pair the sweep to the operation
wandb.agent(sweep_id, function=train)