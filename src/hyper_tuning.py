
import argparse
from ray import tune

from train import train
from utils import load_config

if __name__ == '__main__':
    """Performs hyperparameter tuning for INN and CVAE

    Source: https://docs.ray.io/en/master/tune/

    TODO:
    allow use of cpu and gpu
    add stop metrics
    add metric
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

        config["lr_rate"] = tune.loguniform(0.01, 0.0001)
        # config["batch_size"] = tune.qrandint(100, 3000, 100)
        config["num_layers_subnet"] = tune.qrandint(3, 7, 1)
        config["num_coupling_layers"] = tune.qrandint(4, 8, 1)
        config["hidden_dim"] = tune.qrandint(100, 300, 50)

    elif config["model"] == "CVAE":

        config["lr_rate"] = tune.loguniform(0.01, 0.0001)
        # config["batch_size"] = tune.qrandint(100, 3000, 100)
        config["num_layers"] = tune.qrandint(3, 15, 1)
        config["hidden_dim"] = tune.qrandint(200, 500, 50)

    else:
        raise Exception("Model not supported")

    # enable hyperparameter tuning
    config["hp_tuning"] = True

    analysis = tune.run(
        train,
        config=config,
        num_samples=2,
        resources_per_trial={"gpu": 0, "cpu": 2},
        metric="loss",
        mode="min",
    )
