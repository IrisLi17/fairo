from polymetis import RobotInterface


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="101.6.103.171",
    )

    # Reset
    robot.go_home()