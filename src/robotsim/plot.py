
import numpy as np
import matplotlib.pyplot as plt


def plot(joint_states, robot, title=None, path=None, show=False):
    """Plots robot for specified joint states.
    Also accepts batches of joint states.

    Args:
    joint_states: joint states to plot
    robot: object of RobotSim
    title: title for plot
    path: plot is saved to image if path is provided
    show: show figure

    Returns:
    fig: figure object of plot for further processing
    ax: axes object of plot for further processing

    Examples:
    js = [1, -2.2, 0.4]
    robot = Robot2D3DoF([3, 2, 3])
    fig, ax = robotsim_plot.plot(js, robot)
    """

    joint_coords = robot.get_joint_coords(joint_states)

    fig, ax = plt.subplots()
    ax.grid()
    plot_limit = np.sum(robot.get_length())*1.1
    ax.set_xlim([-plot_limit, plot_limit])
    ax.set_ylim([-plot_limit, plot_limit])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    if title:
        ax.set_title(title)

    # TODO improve performance
    # one call to plot -> add 'None' between arm configurations
    for arm in joint_coords:
        ax.plot(
            arm[:, 0].flatten(),
            arm[:, 1].flatten(),
            linewidth=1,
            c='b')

    ax.scatter(
        joint_coords[:, :, 0].flatten(),
        joint_coords[:, :, 1].flatten(),
        c='r', s=8)

    if path:
        fig.savefig(path, dpi=300)

    if show:
        fig.show()

    return fig, ax


def heatmap(joint_states, robot, ax=None, highlight=None, transparency=None, title=None, path=None, show=False):
    """Plots heatmat of robot for specified joint states.

    Args:
    joint_states: joint states to plot
    robot: object of RobotSim
    ax: axes instance to plot heatmap
    highlight: index of element in joint_states to highlight
    transparency: factor for transparency of each arm 
        (good values are between 0.001 and 0.3)
        if no value is provided, than a value is automatically calculated.
    title: title for plot
    path: plot is saved to image if path is provided
    show: show figure

    Returns:
    fig: figure object of plot for further processing
    ax: axes object of plot for further processing

    Examples:
    js = [1, -2.2, 0.4]
    robot = Robot2D3DoF([3, 2, 3])
    fig, ax = robotsim_plot.heatmap(js, robot)
    """

    joint_coords = robot.get_joint_coords(joint_states)

    existing_ax = bool(ax)

    if not existing_ax:
        fig, ax = plt.subplots()
        ax.grid()
        plot_limit = np.sum(robot.get_length())*1.1
        # ax.set_xlim([-plot_limit, plot_limit])
        ax.set_xlim([0.0, plot_limit])
        ax.set_ylim([-plot_limit/3, plot_limit/3])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    if title:
        ax.set_title(title)

    if not transparency:
        # attempt at automatic transparency
        transparency = np.clip(
            np.exp(-0.008*joint_coords.shape[0]),
            0.01, 1)

    for arm in joint_coords:
        ax.plot(
            arm[:, 0].flatten(),
            arm[:, 1].flatten(),
            c='b', alpha=transparency)

    if highlight is not None:
        arm = joint_coords[highlight]
        ax.plot(
            arm[:, 0].flatten(),
            arm[:, 1].flatten(),
            c='k', linewidth=2, zorder=10)
        ax.scatter(
            arm[:, 0].flatten(),
            arm[:, 1].flatten(),
            c='r', s=6, zorder=11)

    if path and not existing_ax:
        fig.savefig(path, dpi=300)

    if show and not existing_ax:
        fig.show()

    if not existing_ax:
        return fig, ax
    else:
        return ax
