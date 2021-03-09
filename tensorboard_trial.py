import wandb
import tensorflow as tf
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import numpy as np
import sys
import os

# Added matplotlib for debugging data
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

def main():
	path_to_events_file = sys.argv[1]
	for root, dirs, files in os.walk(path_to_events_file):
		for file in files:
			if "tfevents" in file:
				path_to_events_file = os.path.join(root, file)
		
	param_dict = {"steps": 1e8,
	                           "batch_size": 1024,
	                           "gamma": 0.99,
	                           "lambda": 0.95
	                           }

	tf_size_guidance = {
	    'compressedHistograms': 10,
	    'images': 0,
	    'scalars': 1e8,
	    'histograms': 1
	}
        
	event_acc = EventAccumulator(path_to_events_file, tf_size_guidance)
	event_acc.Reload()

	rows = len(event_acc.Scalars('Is Training')) + 1
	columns = len(event_acc.Tags()['scalars']) + 1

	#Create empty dictionary for scalar step and value data
	steps = {}
	values = {}
	#Fetch list of scalars
	scalars = list(event_acc.Tags()['scalars'])

	# Iterate through scalars, and add the relevant components to the dictionary
	for scalar in scalars:
		steps[scalar] = [item.step for item in event_acc.Scalars(scalar)]
		values[scalar] = [item.value for item in event_acc.Scalars(scalar)]
		unique_steps = sorted(set(step for step in steps[scalar]))

	print(scalars)

	wandb.init(project="test-drive-2", config=param_dict)

	for step in unique_steps:
		for scalar in scalars:
			if step in steps[scalar]:
				step_index = steps[scalar].index(step)
				print("Step: ", step, "Scalar: ", scalar, "Step position in dict: ", step_index, "Corresponding value: ", values[scalar][step_index])
				wandb.log({scalar: values[scalar][step_index]}, step=step)

	# wandb.log.run.summary("accuracy"= )


	wandb.finish()
	quit()

if __name__=="__main__":
    main()
