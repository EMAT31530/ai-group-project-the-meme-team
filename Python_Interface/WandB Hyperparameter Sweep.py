#Wandb sign in and set up. Run as separate function/ cell in code.
import wandb
wandb.login

#Multiple types of search are available can be defined as below:
#method can be 'random' or'grid'. remove the 'metric' term to use these.
sweep_config = {
                'method': 'bayes'
                'metric': { #use only for bayesian searches
                            'name': 'loss',
                            'goal': 'minimize'   #if want sle. for accuracy, 'minimize' should be the 'goal'
                            }
                            
                'parameters':{
                    #Hyperparameters taken from config.yaml file, original values commented out
                    'batch_size':{ 
                        'values': [512,5120] } #old 2048   for disc 32,512 cont 512,5120 should always be a fraction of buffer size. cont: 1000s, disc 10s
                    'buffer_size':{
                        'values': [2048, 409600] } #5096
                    'learning_rate':{
                        'values': [1e-5 ,1e-3]} # 0.00025
                    'beta':{
                        'values': [1e-4, 1e-2]} # 0.001
                    'epsilon':{
                        'values': [0.2]} #  0.2
                    'lambd': {
                        'values': {[0.9,0.95]} # 0.99
                    'num_epoch':{ 
                        'values': [3,10]}   #3
                    # learning_rate_schedule: linear
                    #network_settings:1e-3
                    # normalize: true
                    'hidden_units':{
                        'values': [32,512]} # 256
                    'num_layers': {
                        'values': [1,3]}  #2
                    #reward_signals:
                    extrinsic:
                    'gamma':{
                        'values': [0.8, 0.995]} #0.99
                    # 'strength':{
                        # 'values': [1.0]} 
                    #'max_steps': 100000000.0
                    'time_horizon':{
                        'values':[32,2048]} #64
                    #'summary_freq': 50000
                    }   
       
                }
   
  # constant parameters:
   
   strength:1
   max_steps: 1e-9
   summary_freq: 50000
   network_settings: 1e-3
   normalize: true
   learning_rate_schedule: linear


#To intialise the parameter sweep: this should provide a link to the sweep in the browser to use and track the runs.
sweep_id= wandb.sweep(sweep_config)

# The agent then needs to run as the corresponding function naming convention below:

def train():

#initialise wandb for run - V IMPORTANT
wandb.init(project = "rocketlander")

#execute the function to train the agent and pair the sweep to the operation
wandb.agent(sweep_id, function=train)