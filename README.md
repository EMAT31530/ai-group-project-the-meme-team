# Rocket Lander Repository

This is the repository for the Rocket Lander project, as part of EMAT31530 Introduction to Artificial Intelligence.

## Project Objective
The goal is to use Unity to train an AI agent to land a rocket, inspired by the work by SpaceX. This will be facilitated through the use of the PPO (Proximal Policy Optimisation) learning algorithm, implemented through the Unity ML-Agents Package. A custom environment will be used for training and testing, allowing for a detailed investigation into the capabilities of the agent as a result of different environmental settings.

## Installation Instructions
In order to install the required Python libraries, use "pip install -r requirements.txt". These have been tested using Python 3.8.7.

Unity version 2020.2.0f1 has been used to develop the simulation, using the ML Agents (v1.7.2-preview) and Input System (v1.0.1) packages from the Unity Registry.

## File Structure / Usage Information
The Unity simulation environment is located within the "Rocket_Lander" directory, whereas the trained agent files are located within the "Agent_Training" directory. Within the Agent_Training directory, there is a subdirectory for config (.YAML) files and results from the training procedure. All files should ideally be prefixed with a timestamp for traceability. 

A new model can be trained by running the "mlagents-learn config/<CONFIG_NAME>.yaml --run-id=<RUNID_NAME>" command, replacing <CONFIG_NAME> with the name of the target config file, and <RUNID_NAME> with the corresponding name for the model iteration. The Unity environment can then be run, which will begin the training process, and the results will be placed into a folder named <RUNID_NAME>.

Training results can be viewed using Tensorboard by using the "tensorboard --logdir <RUNID_NAME>" command from the "Agent_Training" directory, followed by opening "localhost:6060" using a web browser. An accompanying text file (e.g. "Description.txt") will be located within each result folder to store a summary of experimental settings and interpretation of results.

## Team Members:  
Dan Rodrigues  
Ollie Fogg  
Smruthi Radhakrishnan  
Jake Hodges  
Michael Lynch
