
import argparse
import math
from ray import tune

from train import train
from utils import load_config
from hyperparam_search import *


def stopper(trial_id, result):
    """Stops trial prematurely.
    """

    # don't stop unless training has run for a bit
    if result["epoch"] < 10:
        return False

    # stop if loss is unstable
    if math.isnan(result["loss"]):
        return True
    if result["loss"] > 10000:
        return True

    return False


if __name__ == '__main__':
    """Performs hyperparameter tuning for INN and CVAE

    Source: https://docs.ray.io/en/master/tune/

    """

    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning of neural networks for inverse kinematics.")
    parser.add_argument(
        "config_file",
        help="File containing configurations. "
        "Can be a name of a file in the configs directory "
        "or the path to a config file.")
    args = parser.parse_args()
    config = load_config(args.config_file)

    print("Starting hyperparameter optimization")

    # set hyperparameter search spaces
    # overwrites values in the config file
    # https://docs.ray.io/en/master/tune/api_docs/search_space.html#random-distributions-api
    if config["model"] == "INN":

        config["lr_rate"] = tune.loguniform(0.005, 0.0001)
        # config["batch_size"] = tune.choice([100, 250, 500, 1000])
        config["num_layers_subnet"] = tune.qrandint(3, 7, 1)
        config["num_coupling_layers"] = tune.qrandint(6, 10, 1)
        config["hidden_dim"] = tune.qrandint(100, 300, 50)
        # config["weight_Ly"] = tune.loguniform(0.0001, 1.0)
        # config["weight_Lz"] = tune.loguniform(1.0, 10000)
        # config["weight_Lx"] = tune.loguniform(1.0, 10000)
        # config["weight_Lxy"] = tune.loguniform(0.0001, 1.0)
        config["weight_decay"] = tune.loguniform(0.00001, 0.001)


    elif config["model"] == "CVAE":

        config["lr_rate"] = tune.loguniform(0.01, 0.0001)
        # config["variational_beta"] = tune.loguniform(0.00001, 0.01)
        # config["batch_size"] = tune.choice([100, 250, 500, 1000])
        config["num_layers"] = tune.qrandint(3, 15, 1)
        config["hidden_dim"] = tune.qrandint(200, 500, 50)
        config["weight_decay"] = tune.loguniform(0.00001, 0.001)

    else:
        raise Exception("Model not supported")

    # enable hyperparameter tuning
    config["hyperparam_tuning"] = True

    # disable checkpoints
    # prevents using too much storage space
    config["checkpoint_epoch"] = 0

    analysis = tune.run(
        train,
        metric="loss",
        mode="min",
        config=config,
        num_samples=15,
        # resources can also be fractional values
        # this will determine how many workers
        # are going to run in parallel
        resources_per_trial={"gpu": 0.5, "cpu": 2},
        # stop trial if loss explodes
        stop=stopper
    )

    print("Best configuration:")
    print(analysis.best_config)