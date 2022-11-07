from polymetis import GripperInterface
import time
import sys


if __name__ == "__main__":
    ip = sys.argv[1]
    mode = sys.argv[2]
    assert mode in ["open", "close"]
    gripper = GripperInterface(ip_address=ip)
    # gripper_state = gripper.get_state()
    # print("Gripper state", gripper_state, gripper_state.is_moving)
    if mode == "close":
        gripper.grasp(0.02, 5)
    # [Yunfei] It is important to sleep here, otherwise the next goto command will flush out the previous one
    # time.sleep(1)
    # gripper_state = gripper.get_state()
    # print("Gripper state", gripper_state, gripper_state.is_moving)
    elif mode == "open":
        gripper.goto(0.08, 0.05, 1)