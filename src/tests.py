
import numpy as np

from robotsim import RobotSim2D


# robot = RobotSim2D(1, 3)
# print(robot.forward(0))

# robot = RobotSim2D(1, [3])
# print(robot.forward([0]))

# robot = RobotSim2D(2, [3, 2])
# print(robot.forward([0, 1]))

# robot = RobotSim2D(2, [3, 2])
# print(robot.forward([0, 1]))

# robot = RobotSim2D(3, [3, 2, 1])
# theta = [-np.pi/4, 0, 0]
# x = robot.forward(theta)

# robot.plot_configurations(theta)

robot = RobotSim2D(3, [3, 2, 1])
theta = [-np.pi/4, 0, 0]

x = robot.forward(theta)
robot.plot_configurations(theta)

theta_inv = robot.inverse(x)