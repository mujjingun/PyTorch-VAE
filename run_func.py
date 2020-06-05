import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer

def insert(node, dic):
    for key, value in dic.items():
        if isinstance(value, dict):
            if key not in node:
                node[key] = dict()
            insert(node[key], value)
        else:
            node[key] = value

def run_func(config, **kwargs):
    config = yaml.safe_load(config)

    insert(config, kwargs)

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                            config['exp_params'])

    runner = Trainer(min_nb_epochs=1,
                    train_percent_check=config['exp_params']['fraction'],
                    val_percent_check=1.,
                    num_sanity_val_steps=0,
                    early_stop_callback=False,
                    checkpoint_callback=False,
                    logger=False,
                    weights_summary=None,
                    **config['trainer_params'])
    runner.fit(experiment)

    print("validation loss = ", experiment.val_loss.item())
    return experiment.val_loss.item()
