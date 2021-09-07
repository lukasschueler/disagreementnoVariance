import wandb
import random

random.seed(1)

# Initalize a new run
wandb.init(project="testingCenter", group = "custom axes", entity = "lukischueler")

# Define the custom x axis metric
wandb.define_metric("custom_step")
wandb.define_metric("custom_step23")

# Define which metrics to plot against that x-axis
wandb.define_metric("Reward", step_metric='custom_step')
wandb.define_metric("Length of Episode", step_metric='custom_step')

wandb.define_metric("Reward", step_metric='custom_step23')
wandb.define_metric("Length of Episode", step_metric='custom_step23')


# for i in range(4):
#   log_dict = {
#       "custom_step": i,   
#   }
#   wandb.log(log_dict)
  
  
wandb.log({
    "Reward": 3,
    "custom_step23": 4
})
wandb.log({
    "Length of Epsiode": 10,
    "custom_step": 1
})

wandb.log({
    "Reward": 13,
    "custom_step": 1
})

wandb.log({
    "Length of Epsiode": 6,
    "custom_step23": 5,
    
    "custom_step": 5
})
wandb.log({
    "Reward": 25,
    "custom_step": 2
})

wandb.log({
    "Length of Epsiode": 6,
    "custom_step": 2
})

wandb.log({
    "Length of Epsiode": 17,
    "custom_step": 3
})

wandb.log({
    "Length of Epsiode": 17,
    "custom_step": 13
})

wandb.log({
    "Length of Epsiode": 0,
    "custom_step": 13
})
# Use this in the context of a jupyter notebook to mark a run finished
wandb.finish()