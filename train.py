import wandb
import yaml
from hyperbolic import model

def train():
    with wandb.init():
        config = wandb.config
        model(config)


if __name__ == "__main__":
    with open(r'sweep.yaml') as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep_config, project="HPCS")
    wandb.agent(sweep_id, function=train, count=1, project="HPCS")