from polymetis import RobotInterface
import torch


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="101.6.103.171",
    )
    # Ready pose in ROS
    robot.set_home_pose(
        torch.Tensor([0., -0.785, 0., -2.356, 0., 1.571, 0.785])
    )
    # Reset
    robot.go_home()