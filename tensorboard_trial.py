import wandb
import tensorflow as tf
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import numpy as np
import sys

# Added matplotlib for debugging data
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

def main():
        #path_to_events_file = sys.argv[1]

        # Temporary hardcoded path for ease of debug
        path_to_events_file = r"Agent_Training\results\2021.02.10_1700_RocketLander_SR\RocketLander\events.out.tfevents.1612976372.DESKTOP-6RE48I1.13812.0"

        learning_rate = 0
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




        # Create empty dictionary for scalar step and value data
        steps = {}
        values = {}

        # Fetch list of scalars
        scalars = list(event_acc.Tags()['scalars'])

        # Iterate through scalars, and add the relevant components to the dictionary
        for scalar in scalars:
                steps[scalar] = [item.step for item in event_acc.Scalars(scalar)]
                values[scalar] = [item.value for item in event_acc.Scalars(scalar)]

        # Select the data pertaining to one scalar
        #s = steps['Policy/Entropy']
        #v = values['Policy/Entropy']
        s = steps['Policy/Learning Rate']
        v = values['Policy/Learning Rate']

        # Plot and show
        plt.plot(s, v)
        plt.show()



        
        #Create empty 2-D array for parsing log data into
        int_data = np.empty((rows,columns),int)
        #Empty array of object datatype to add category strings
        cat_data= np.empty((rows,columns),object)

        categories_list = event_acc.Tags()['scalars']
    
        for cat_index,category in enumerate(categories_list):
                print(cat_index, category)
                cat_data[0,cat_index+1]=category
                for value_index, s in enumerate(event_acc.Scalars(category)):
                        value_temp = s.value
                        step_temp = s.step
                        int_data[value_index+1,0]= step_temp
                        int_data[value_index+1,cat_index+1]=value_temp

        # Sort array by ascending order of step to avoid wandb error
        int_data = int_data[np.argsort(int_data[:, 0])]
        cat_data[1:,:]=int_data[1:,:]
        
        wandb.init(project="test-drive-2", config=param_dict)
        for row in range(1,rows):
                for column in range(1,columns):
                        category = cat_data[0,column]
                        if(cat_data[row,column]==None):
                                log_value=None
                        else:
                                log_value=float(cat_data[row,column])
                        log_step=cat_data[row,0]
                        wandb.log({category: log_value}, step=log_step)

        wandb.finish()
        quit()

if __name__=="__main__":
        main()
