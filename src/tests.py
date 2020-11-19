
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

robot = RobotSim2D(3, [2, 2, 2])

theta = [-np.pi/4, 0, 0]
# theta = [-np.pi/4, np.pi/2, np.pi/4]

# print('theta: ', theta)
x = robot.forward(theta)

# print('TCP coordinates: ', x)
robot.plot_configurations(theta)

