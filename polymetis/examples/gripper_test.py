from polymetis import GripperInterface
import time


if __name__ == "__main__":
    gripper = GripperInterface(ip_address="101.6.103.171")
    # gripper_state = gripper.get_state()
    # print("Gripper state", gripper_state, gripper_state.is_moving)
    # gripper.grasp(0.02, 1)
    # [Yunfei] It is important to sleep here, otherwise the next goto command will flush out the previous one
    # time.sleep(1)
    # gripper_state = gripper.get_state()
    # print("Gripper state", gripper_state, gripper_state.is_moving)
    gripper.goto(0.08, 0.05, 1)