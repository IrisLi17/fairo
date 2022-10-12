from collections import deque
from polymetis import RobotInterface, GripperInterface, CameraInterface
import polymetis_pb2
import termios, tty, sys
import queue
import threading
import time
import torch
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torchcontrol.transform.rotation as rotation
import pygame


class DemoCollector:
    VALID_KEYS = ["a", "w", "s", "d", "i", "k", "space", "esc"]
    KEY_MAPPINGS = {
        32: "space",
        27: "esc"
    }
    def __init__(self, ip_address="localhost"):
        self.robot = RobotInterface(
            ip_address=ip_address
        )
        self.gripper = GripperInterface(
            ip_address=ip_address
        )
        self.camera = CameraInterface(
            ip_address=ip_address
        )
        self._image_buffer = deque(maxlen=1)

        # self._keyboard_thr = threading.Thread(
        #     target=self._keyboard_listener,
        #     daemon=True,
        # )
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self._control_thr = threading.Thread(
            target=self._joystick_listener, daemon=True
        )

    def run(self, demo_path="demo.pkl"):
        if os.path.exists(demo_path):
            ans = input(demo_path + " exists, going to remove? [y]")
            if ans == "y":
                os.remove(demo_path)
            else:
                return
        self.robot.go_home()
        self.robot_initial_quat = self.robot.get_ee_pose()[1]
        self.robot.start_cartesian_impedance()
        # self._keyboard_thr.start()
        self._control_thr.start()
        # self._command_thr.start()
        fig, ax = plt.subplots(1, 1)
        print("Listening keyboard events")
        while True:
            image, timestamp = self.camera.read_once()
            self._image_buffer.append((image, timestamp))
            ax.cla()
            ax.imshow(image.astype(np.uint8))
            ax.set_title(timestamp)
            plt.pause(0.01)
        
    def _joystick_listener(self):
        eef_quat = (rotation.from_quat(torch.Tensor([0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)])) * \
            rotation.from_quat(torch.Tensor([1.0, 0, 0, 0]))).as_quat()
        print("EEF quat", eef_quat)
        while True:
            dx, dy, dz = 0., 0., 0.
            gripper_open, gripper_close = False, False
            events = pygame.event.get([pygame.JOYAXISMOTION, pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION])
            if len(events) == 0:
                continue
            record_obj = dict()
            robot_state = self.robot.get_robot_state()
            gripper_state = self.gripper.get_state()
            try:
                camera_image, timestamp = self._image_buffer.pop()
            except:
                continue
            eef_pos, init_eef_quat = self.robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions))    
            for event in events:
                if event.type == pygame.JOYAXISMOTION:
                    cur_value = event.value if abs(event.value) > 0.5 else 0
                    if event.axis == 3:
                        if event.value > 0:
                            dy = max(dy, 0.01 * cur_value)
                        else:
                            dy = min(dy, 0.01 * cur_value)
                    elif event.axis == 4:
                        if event.value > 0:
                            dx = max(dx, 0.01 * cur_value)
                        else:
                            dx = min(dx, 0.01 * cur_value)
                    elif event.axis == 1:
                        if event.value > 0:
                            dz = max(dz, 0.01 * cur_value)
                        else:
                            dz = min(dz, 0.01 * cur_value)
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 4:
                        if not gripper_state.is_moving:
                            if gripper_state.is_grasped:
                                gripper_open = True
                            else:
                                gripper_close = True
            if gripper_open:
                self.gripper.goto(0.08, 0.1, 1)
                record_obj["desired_gripper"] = "open"
            elif gripper_close:
                self.gripper.grasp(0.1, 1)
                record_obj["desired_gripper"] = "close"
            elif abs(dx) > 0 or abs(dy) > 0 or abs(dz) > 0:
                eef_pos = eef_pos + torch.Tensor([dx, dy, dz])
                self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
            else:
                # No robot motion
                continue
            # Save
            with open("demo.pkl", "ab") as f:
                record_obj.update(dict(
                    robot_state=robot_state, gripper_state=gripper_state, image=camera_image.astype(np.uint8), 
                    desired_eef_pos=eef_pos, desired_eef_quat=eef_quat
                ))
                pickle.dump(record_obj, f)
                print("Demo saved")

    def _keyboard_listener(self):
        eef_quat = (rotation.from_quat(torch.Tensor([0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)])) * \
            rotation.from_quat(torch.Tensor([1.0, 0, 0, 0]))).as_quat()
        print("EEF quat", eef_quat)
        while True:
            keyname = self._getkey()
            print(keyname, "pressed")
            if keyname in self.VALID_KEYS:
                record_obj = dict()
                robot_state = self.robot.get_robot_state()
                gripper_state = self.gripper.get_state()
                try:
                    camera_image, timestamp = self._image_buffer.pop()
                except:
                    continue
                eef_pos, init_eef_quat = self.robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions))
                # Parse rpy from eef_quat, roll = 180deg, pitch = 0deg, yaw=?
                # eef_roll, eef_pitch, eef_yaw = quat2euler(eef_quat)
                # eef_quat = euler2quat(torch.Tensor([torch.pi, 0., eef_yaw]))
                if keyname == "a":
                    eef_pos = eef_pos + torch.Tensor([0.01, 0.0, 0.0])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif keyname == "d":
                    eef_pos = eef_pos + torch.Tensor([-0.01, 0.0, 0.0])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif keyname == "w":
                    eef_pos = eef_pos + torch.Tensor([0.0, 0.01, 0.0])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif keyname == "s":
                    eef_pos = eef_pos + torch.Tensor([0.0, -0.01, 0.0])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif keyname == "i":
                    eef_pos = eef_pos + torch.Tensor([0.0, 0.0, 0.01])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif keyname == "k":
                    eef_pos = eef_pos + torch.Tensor([0.0, 0.0, -0.01])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                # elif keyname == ",":
                #     eef_quat = euler2quat(torch.Tensor([torch.pi, 0, eef_yaw + 0.1 * torch.pi]))
                #     self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif keyname == "space":
                    if not gripper_state.is_moving:
                        if gripper_state.is_grasped:
                            self.gripper.goto(0.08, 0.1, 1)
                            record_obj["desired_gripper"] = "open"
                        else:
                            self.gripper.grasp(0.1, 1)
                            record_obj["desired_gripper"] = "close"
                elif keyname == "esc":
                    os.system("stty sane")
                    print("Exit keyboard listener")
                    break
                # Save
                with open("demo.pkl", "ab") as f:
                    record_obj.update(dict(
                        robot_state=robot_state, gripper_state=gripper_state, image=camera_image.astype(np.uint8), 
                        desired_eef_pos=eef_pos, desired_eef_quat=eef_quat
                    ))
                    pickle.dump(record_obj, f)
                    print("Demo saved")

    def _getkey(self):
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        try:
            while True:
                b = os.read(sys.stdin.fileno(), 3).decode()
                if len(b) == 3:
                    k = ord(b[2])
                else:
                    k = ord(b)
                return self.KEY_MAPPINGS.get(k, chr(k))
        except:
            return None
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/5_collect_demo.py [ip]")
    ip = sys.argv[1]
    demo_controller = DemoCollector(ip_address=ip)
    demo_controller.run()
