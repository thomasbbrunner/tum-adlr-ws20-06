import numpy as np
import pdb

from robotsim import RobotSim2D
from robotsim_dataset import RobotSimDataset

# robotsim.py

robot = RobotSim2D(2, [3, 2])
print(robot.forward([0, 1]))
print(robot.forward([[0, 1], [1, 0], [1, 1]]))
print(robot.inverse([[1, 1, 1], [1, 1, 0]]))

robot.plot_configurations([0, 1])
robot.plot_configurations([[0, 1], [1, 0]])

robot = RobotSim2D(3, [3, 3, 3])
print(robot.forward([0, 1, 1]))
print(robot.forward([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
print(robot.inverse([3, -4, 0]))
# [[-2.411865    1.68213734  0.72972766]
#  [-0.72972766 -1.68213734  2.411865  ]]
print(robot.inverse([[2, 2, 2], [5, 1, 0.3], [50, 1, 0.3]]))
# [[[-1.2030684   1.9652703   1.2377981 ]
#   [ 0.7622019  -1.9652703   3.2030684 ]]
#  [[-1.15352504  2.41326677 -0.95974173]
#   [ 1.25974173 -2.41326677  1.45352504]]
#  [[        nan         nan         nan]
#   [        nan         nan         nan]]]
robot.plot_configurations([1, 2, 3])
robot.plot_configurations([[1, 2, 3], [-3, -2, -1]])

# dataset.py

robot = RobotSim2D(3, [6, 7, 3])
dataset = RobotSimDataset(robot, 100)
