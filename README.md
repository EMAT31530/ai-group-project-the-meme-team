# Rocket Lander Repository

This is the repository for the Rocket Lander project, as part of EMAT31530 Introduction to Artificial Intelligence.

## Project Objective
The goal is to train an AI agent to land a thrust-vector controlled rocket, inspired by the work done by SpaceX. A custom environment has been developed using Unity, and the PPO (Proximal Policy Optimisation) learning algorithm has been used to try and achieve this goal, implemented through the ML-Agents Package. 

## Installation Instructions
In order to install the required Python libraries, use "pip install -r requirements.txt". These have been tested using Python 3.8.7. Pytorch is additionally required, which can be downloaded by following the instructions at https://pytorch.org/get-started/locally/.  

Unity version 2020.2.0f1 has been used to develop the simulation, using the ML Agents (v1.7.2-preview) and Input System (v1.0.1) packages from the Unity Registry. This will be required to make changes to the environment, and to build the project to the Python_Interface/Unity_Compiled_Files folder, although training (and even evaluation) can be facilitated through this built environment following this.   

## File Structure / Usage Information
The Unity simulation environment is located within the "Rocket_Lander" directory, with a self-explanatory folder structure. One script is used to control all behaviours, titled "Rocket_Agent.cs". The Networks folder is used to store the trained neural networks (.onnx). These can be dragged into the Unity editor to add them to the project, then assigned to a rocket by selecting the relevant neural network model from within the "Rocket_Agent (Script)/Behaviour Parameters/Model" field within the Simulation_Environment/Rocket Game Object.  

Model training is abstracted through the "Model_Configurator_Manual.py" and "Model_Configurator_Sweep.py" file within the Python_Interface folder. The Model_Configurator_Manual.py file provides an interface to run a custom model with user-defined arguments, which is ideal for finetuning parameters, or conducting a longer run. In contrast, the Model_Configurator_Sweep.py file interfaces with Weights and Biases to perform a parameter sweep, and so is preconfigured with a smaller step count (3e7).

Upon running either file, a corresponding config file will be created, followed by launching the Unity ML-Agents trainer. This will point to the compiled environment within the Unity_Compiled_Files folder, hence the project must be built (in the Python_Interface/Unity_Compiled_Files directory) prior to running the script. A built version has been included with the repository, however if changes are required, it must be rebuilt in the same target location. Upon completion of training, the file structure will be renamed to our prefered file structure for consistent datetime formatting. 

Training results can also be viewed using Tensorboard by using the "tensorboard --logdir results" command from the Python_Interface directory, followed by opening "localhost:6060" using a web browser. This enable live performance monitoring, and simultaneous viewing of all training processes (if targeted root results folder). Alternatively, results can be monitored through the Weights and Biases platform.

### Weights and Biases Set-Up Instructions (First-Time Operation)
When using weights and biases for the first time:
1) Run: pip install --upgrade wandb
2) Login to weights and biases (https://wandb.ai/login) and copy API authorisation code from: https://wandb.ai/authorize
3) Run: wandb login *Your authorisation code*

### Weights and Biases Training Instructions
Make sure your local Model_Configurator_Sweep.py script is up to date with the remote version. From within the Python_Interface directory, run the following command:

wandb agent --count 1 uob_rocket_lander/rocketlander/w34jrc7s

Note: --count specifies the number of trainings to run, so change/ remove this for the agent to automatically run multiple trainings 

## Team Members:  
Dan Rodrigues  
Ollie Fogg  
Smruthi Radhakrishnan  
Jake Hodges  
Michael Lynch
