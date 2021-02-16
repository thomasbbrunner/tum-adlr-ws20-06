
from matplotlib.figure import Figure
import pathlib
from ray import tune
import utils

CONFIG_PLOT = {
    "lr_rate": "log",
    "hidden_dim": "linear",
    # cVAE
    "num_layers": "linear",
    # INN
    "num_coupling_layers": "linear",
    "num_layers_subnet": "linear",
}


def generate_plots(analysis, path):
    """Generate plots of the results of a hyperparameter search.

    Args:
    analysis: either an instance of ExperimentAnalysis or Analysis
    """
    path = pathlib.Path(path)
    dataframe = analysis.dataframe()

    loss = dataframe["loss"]
    figures = []

    for key in CONFIG_PLOT.keys():

        name = "config/" + key
        if name not in dataframe.keys():
            continue

        fig = Figure()
        ax = fig.add_subplot()
        ax.scatter(dataframe[name], loss)
        ax.grid()
        ax.set_title("{}".format(key))
        ax.set_xlabel("{}".format(key))
        ax.set_xscale(CONFIG_PLOT[key])
        ax.set_ylabel("loss")
        ax.set_ylim([0, 10])
        fig.savefig(path/key)
        figures.append(fig)

    return figures


def get_best(analysis):

    df = analysis.trial_dataframes

    best_final = {}
    for key, value in df.items():
        best_final[key] = value.at[value.index[-1], "loss"]

    best_any = {}
    for key, value in df.items():
        best_any[key] = value["loss"].min()

    # sort best
    best_final = list(sorted(best_final.items(), key=lambda item: item[1]))
    best_any = list(sorted(best_any.items(), key=lambda item: item[1]))

    print("Configurations with smallest final loss:")
    for i in range(10):
        print("  {}) '{}'   ({})"
              .format(i+1, best_final[i][0], best_final[i][1]))

    print("Configurations with smallest loss (any episode):")
    for i in range(10):
        print("  {}) '{}'   ({})"
              .format(i+1, best_any[i][0], best_any[i][1]))

    return best_final, best_any


if __name__ == "__main__":

    analysis = tune.Analysis(
        "./results/train_2021-02-14_11-12-08",
        "loss", "min")

    # make sure results directory exists
    directory = pathlib.Path("results")
    directory.mkdir(exist_ok=True)

    figures = generate_plots(analysis, directory)
    best = get_best(analysis)
