#Wandb sign in and set up. Run as separate function/ cell in code.
import wandb
#wandb.login

#Multiple types of search are available can be defined as below:
#method can be 'random' or'grid'. remove the 'metric' term to use these.
sweep_config = {
                'method': 'bayes',
                
                'metric': { #use only for bayesian searches
                            'name': 'loss',
                            'goal': 'minimize'   #if want sle. for accuracy, 'minimize' should be the 'goal'
                            },
                            
                'parameters':{
                    #Hyperparameters taken from config.yaml file, original values commented out
                    'batch_size':{ 
                        'values': [512,1024,2048,4096,5120] }, #old 2048   for disc 32,512 cont 512,5120 should always be a fraction of buffer size. cont: 1000s, disc 10s
                    'buffer_size':{
                        'values': [2048,5096,10192,2038,40768,81536,163072,326144, 409600] }, #5096
                    'learning_rate':{
                        'distribution':'uniform',
                        'min':1e-5 ,
                        'max': 1e-3}, # 0.00025
                    'beta':{
                        'distribution':'uniform',
                        'min': 1e-4,
                        'max': 1e-2}, # 0.001
                    'epsilon':{
                        'distribution':'uniform',
                        'min': 0.1,
                        'max': 0.5}, #  0.2
                    'lambda': {
                        'distribution':'uniform',
                        'min': 0.9,
                        'max': 0.95}, # 0.99
                    'num_epoch':{ 
                        'values': [3,4,5,6,7,8,9,10]},   #3
                   
                    'hidden_units':{
                        'distribution':'uniform',
                        'min': 32,
                        'max':512}, # 256
                    'num_layers': {
                        'values': [1,2,3]},  #2 
                    'gamma':{
                        'distribution':'uniform',
                        'min': 0.8, 
                        'max': 0.995}, #0.99
                    'time_horizon':{
                        'distribution':'uniform',
                        'min':32,
                        'max':2048}, #64
                        
                    # constant parameters:
                    'strength':1,
                    'max_steps': 1e-9,
                    'summary_freq': 50000,
                    'network_settings': 1e-3,
                    'normalize': True,
                    'learning_rate_schedule': 'linear',
                    'network_settings':1e-3,
                                }   
       
            }


wandb.init(name='Rocket_Lander_2021.03.03_1234_JAH', 
           project='rocketlander',
           notes='This is a test run 2', 
           tags=['RocketLander', 'Test Run'],
           entity='uob_rocket_lander',
           )

               
#To intialise the parameter sweep: this should provide a link to the sweep in the browser to use and track the runs.
sweep_id= wandb.sweep(sweep_config)


#execute the function to train the agent and pair the sweep to the operation
#wandb.agent(sweep_id, function=train)
