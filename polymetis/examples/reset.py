from polymetis import RobotInterface
import torch
import sys


if __name__ == "__main__":
    # Initialize robot interface
    ip = sys.argv[1]
    robot = RobotInterface(
        ip_address=ip,
    )
    # Ready pose in ROS
    # robot.set_home_pose(
    #     torch.Tensor([0., -0.785, 0., -2.356, 0., 1.571, 0.785])
    # )
    # Reset
    robot.go_home()