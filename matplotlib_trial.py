import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb

def main():
	# path_to_events_file = "Agent_Training\results\2021.02.10_1700_RocketLander_SR\RocketLander\events.out.tfevents.1612976372.DESKTOP-6RE48I1.13812.0"
	path_to_events_file = sys.argv[1]
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

	        
   # Create empty dictionary for scalar step and value data
	steps = {}
	values = {}

    # Fetch list of scalars
	scalars = list(event_acc.Tags()['scalars'])
	i = 1

	wandb.init(project="test-drive-2", config=param_dict)

	# steps 

	# for scalar in scalars:
	# 	steps[scalar] = [item.step for item in event_acc.Scalars(scalar)]
	# 	values[scalar] = [item.value for item in event_acc.Scalars(scalar)]
	# 	fig = plt.figure(i)
	# 	plt.plot(values[scalar])
	# 	plt.ylabel("value")
	# 	plt.xlabel("step")
	# 	fig.suptitle(str(scalar))
	# 	i=i+1
	# 	wandb.log({str(scalar):plt})
	# # plt.show()

	plt.plot([1, 2, 3, 4])
	plt.ylabel('some interesting numbers')
	wandb.log({"chart": plt})

	wandb.finish()


if __name__=="__main__":
        main()



