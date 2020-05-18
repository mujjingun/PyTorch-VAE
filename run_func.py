import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

def insert(node, dic):
    for key, value in dic.items():
        if isinstance(value, dict):
            if key not in node:
                node[key] = dict()
            insert(node[key], value)
        else:
            node[key] = value

def run_func(**kwargs):
    with open('configs/bbvae.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    insert(config, kwargs)

    print(config)

    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                            config['exp_params'])

    runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                    min_nb_epochs=1,
                    logger=tt_logger,
                    log_save_interval=100,
                    train_percent_check=config['exp_params']['fraction'],
                    val_percent_check=1.,
                    num_sanity_val_steps=5,
                    early_stop_callback = False,
                    **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)

    print("======= Evaluating Trained Model =======")
    losses = []
    for batch in experiment.val_dataloader()[0]:
        real_img, labels = batch
        results = model.forward(real_img, labels = labels)
        val_loss = model.loss_function(*results, M_N = 1)
        losses.append(val_loss["loss"])
    return np.mean(losses)