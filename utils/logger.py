#Weights and Biases logger integration
import wandb


class Logger:
    def __init__(self, projectName):
        wandb.init(project=projectName)
    
    def config(self, **kwargs):
        wandb.config = kwargs

    def log(self, **kwargs):
        wandb.log(kwargs)
    
    def watch(self, model):
        wandb.watch(model)


        