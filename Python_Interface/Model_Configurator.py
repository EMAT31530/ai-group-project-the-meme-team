"""
Model_Configurator.py
Description: Simple utility script for configuring and launching training process from python.
"""

# Import dependencies
import os
import subprocess
import glob 
from datetime import datetime
from socket import socket
import wandb


#wandb.login

sweep_config = {
                'method': 'bayes',
                
                'metric': { #use only for bayesian searches
                            'name': 'loss',
                            'goal': 'minimize'   #if want sle. for accuracy, 'minimize' should be the 'goal'
                            },
                            
                'parameters':{
                    #Hyperparameters taken from config.yaml file, original initial values commented out at each step
                    # 'batch_size':{ 
                        # 'values': [512,1024,2048,4096,5120] }, #old 2048   for disc 32,512 cont 512,5120 should always be a fraction of buffer size. cont: 1000s, disc 10s
                    # 'buffer_size':{
                        # 'values': [2048,5096,10192,2038,40768,81536,163072,326144, 409600] }, #5096
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
                    # 'num_epoch':{ 
                        # 'values': [3,4,5,6,7,8,9,10]},   #3
                   
                    # 'hidden_units':{
                        # 'distribution':'uniform',
                        # 'min': 32,
                        # 'max':512}, # 256
                    # 'num_layers': {
                        # 'values': [1,2,3]},  #2 
                    'gamma':{
                        'distribution':'uniform',
                        'min': 0.8, 
                        'max': 0.995}, #0.99
                    # 'time_horizon':{
                        # 'distribution':'uniform',
                        # 'min':32,
                        # 'max':2048}, #64
                        
                    # constant parameters:
                    # 'strength':1,
                    # 'max_steps': 1e-9,
                    # 'summary_freq': 50000,
                    # 'network_settings': 1e-3,
                    # 'normalize': True,
                    # 'learning_rate_schedule': 'linear',
                    # 'network_settings':1e-3,
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

# Wrapped training / execution logic within function
# This is to allow for call from WandB agent
def training_cycle():

    """
    ENVIRONMENT SETUP==========================================================================================

    env: Filepath to the compiled environment
    run_id: The name of the run, default = <TIMESTAMP>_RocketLander
    display_size: The height and width (px) of the displayed window
                  (requires no_graphics = False), default = 500
    seed: The random seed used when training the agent, default = 0
    no_graphics: Set True to skip rendering environment, defalt = False
    force: Set True to override existing files, default = False

    ==========================================================================================================
    """

    # Configure environment
    env = "Unity_Compiled_Files\Rocket_Lander"
    TIMESTAMP = datetime.now().strftime("%Y.%m.%d_%H%M")+"_"
    run_id = TIMESTAMP+"RocketLander"
    display_size = 500
    seed = 0
    no_graphics = False
    force = False

    # Get available port for Unity
    port = ""
    with socket() as s:
        s.bind(('',0))
        port = s.getsockname()[1]


    """
    HYPERPARAMETERS===========================================================================================

    batch_size: The number of samples to be processed per iteration (factor of buffer_size)
    buffer_size: The number of samples to be processed before model update
    learning_rate: The learning rate (the size of the update steps)
    beta: KL Penalty coefficient(to achieve desired level of KL divergence per update; not so important)
    epsilon: PPO loss clipping parameter (defining threshold for objective improvement)
    lambd: Smoothing term for reducing variance in GAE
    num_epoch: The number of times to iterate through whole training dataset
    learning_rate_schedule: How to vary LR as progresses => "constant" or "linear"
    normalize: Whether to normalise the network values => "true" or "false"
    hidden_units: The number of hidden units in the hidden layer(s)
    num_layers: The number of hidden layers in the network
    gamma: Discount factor to prioritise immediate reward (in GAE)
    strength: Quantity by which to multiply and scale the reward given by the environment
    max_steps: The number of steps to train the model on (how long to train)
    time_horizon: How many timesteps to consider when associating a reward with actions
    summary_freq: How often to output summary statistics to log files
    keep_checkpoints: How many checkpoints to keep => can use this to extract ONNX by running
                      mlagents-learn config\<TARGET_CONFIG> --resume, then quit with Ctrl+C (or Esc in GUI)
                      NOTE: resume loads the latest checkpoint, so hide / move files as appropriate
    checkpoint_interval: The interval at which new checkpoints are created

    curriculum: Whether to use curriculum learning (vary parameter and complexity over time)
    target_sizes: List of target sizes (decreasing floats, e.g [10.0, 3.0]) to be used for each lesson
    target_step_fraction: Fraction through training to decrease step (length one less than target_sizes)
    min_lesson_length: Integer specifying the minimum length of a lesson

    ==========================================================================================================
    """

    # WandB Hyperparameter Optimisation config might go here?

    # Core hyperparameters
    batch_size = 2048
    buffer_size = 5096
    # learning_rate = 2.5e-4
    # beta = 1.0e-3
    # epsilon = 0.4
    # lambd = 0.99
    num_epoch = 3
    hidden_units = 256
    num_layers = 2
    # gamma = 0.99
    strength = 1.0
    max_steps = 1e8
    time_horizon = 64


    # Additional hyperparameters
    learning_rate_schedule = "linear"
    normalize = "true"
    summary_freq = 10000
    keep_checkpoints = 5
    checkpoint_interval = 50000

    # Configure curriculum learning
    curriculum = True
    target_sizes = [11.0, 9.0, 7.0, 5.0]
    target_step_fraction = [0.3, 0.5, 0.7]
    min_lesson_length = 100


    # Get current working directory, and config directory
    cwd = os.getcwd()
    config_path = os.path.join(cwd, "config")
    config_file = os.path.join(config_path,run_id+".yaml")

    # Make config directory if doesn't exist
    if not os.path.exists(config_path):
        os.mkdir("config")

    # Change to config directory
    os.chdir(config_path)

    # Write YAML contents to config file
    with open(run_id+".yaml","w") as file:
        file.write(f"""behaviors:
      RocketLander:
        trainer_type: ppo
        hyperparameters:
          batch_size: {batch_size}
          buffer_size: {buffer_size}
          learning_rate: {learning_rate}
          beta: {beta}
          epsilon: {epsilon}
          lambd: {lambd}
          num_epoch: {num_epoch}
          learning_rate_schedule: {learning_rate_schedule}
        network_settings:
          normalize: {normalize}
          hidden_units: {hidden_units}
          num_layers: {num_layers}
        reward_signals:
          extrinsic:
            gamma: {gamma}
            strength: {strength}
        max_steps: {max_steps}
        time_horizon: {time_horizon}
        summary_freq: {summary_freq}
        keep_checkpoints: {keep_checkpoints}
        checkpoint_interval: {checkpoint_interval}\n
    """)

        # Write curriculum information if relevant
        if curriculum:
            file.write("""environment_parameters:
      target_size:
          curriculum:\n""")

            for i in range(len(target_sizes)-1):
                file.write(f"""          - name: Lesson{i+1}
                completion_criteria:
                  measure: progress
                  behavior: RocketLander
                  signal_smoothing: true
                  min_lesson_length: {min_lesson_length}
                  threshold: {target_step_fraction[i]}
                  {"require_reset: false" if i!=0 else ""}
                value: {target_sizes[i]}\n""")

            file.write(f"""          - name: Lesson_{len(target_sizes)+1}
                value: {target_sizes[-1]}""")


    # Change back to default directory
    os.chdir(cwd)

    # Initialise flag to track outcome of ML-Agent training operation
    success_flag = None
    try:

        # Initialise command array for running job
        commands = ["mlagents-learn", config_file, "--env="+env, "--run-id="+run_id,
                    "--seed="+str(seed), "--train", "--base-port="+str(port)]

        # Set display size if parsed
        if bool(display_size):
            commands.extend(["--height="+str(display_size), "--width="+str(display_size)])

        # Set headless if argument enabled
        if no_graphics:
            commands.append("--no-graphics")

        # Enable overwrite of files if enabled
        if force:
            commands.append("--force")

        # Run the command to submit for training
        subprocess.check_output(commands, shell=True)
        success_flag = True

    # Capture crash event, if appropriate (can post-process separately)
    except Exception as e:
        success_flag = False
        #print(e)


    # Change to target results folder
    os.chdir(os.path.join("results", run_id))

    # Fetch all checkpoint and onnx files (recursively)
    files = []
    for types in ('**\*.onnx', '**\*.pt'):
        files.extend(glob.glob(types, recursive=True))

    # Rename files to fit with our datestamped naming convention
    for name in files:
        os.rename(os.path.join(os.getcwd(), name),
                  os.path.join(os.getcwd(), "\\".join(name.split("\\")[:-1]), TIMESTAMP+name.split("\\")[-1]))




    # Execute further logic (pending tensorboard -> wandb.ai script)
    # Idea is to load tensorboard data, manipulate it, then iterate though
    # as if in a training loop for the purpose of transfering to wandb.ai
    
    # print("Process completed: ", success_flag)

    # WandB performance logging might go here



if __name__ == "__main__":
    #training_cycle()
    
    #execute the function to train the agent and pair the sweep to the operation
    wandb.agent(sweep_id, function=training_cycle)




