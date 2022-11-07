import argparse
from collections import deque
from polymetis import RobotInterface, GripperInterface, CameraInterface
import polymetis_pb2
import grpc
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
import cv2
from datetime import datetime


class DemoCollector:
    VALID_KEYS = ["a", "w", "s", "d", "i", "k", "space", "esc"]
    KEY_MAPPINGS = {
        32: "space",
        27: "esc"
    }
    def __init__(self, ip_address="localhost", save_file=True, save_path=None):
        self.robot = RobotInterface(
            ip_address=ip_address
        )
        self.gripper = GripperInterface(
            ip_address=ip_address
        )
        self.camera = CameraInterface(
            ip_address=ip_address
        )
        
        self.save_file = save_file
        self.save_path = save_path

        self._image_buffer = deque(maxlen=1)
        self.read_image_lock = False
        self._camera_thr = threading.Thread(
            target=self._camera_listener, daemon=True
        )

        self.start_or_pause = False # True: start, False: pause
        self.stopped = False
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self._joystick_thr = threading.Thread(
            target=self._joystick_listener, daemon=True
        )
        self._command_queue = queue.Queue(maxsize=1)

        if self.save_file:
            assert self.save_path is not None
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

    def run(self):
        # if os.path.exists(demo_path):
        #     ans = input(demo_path + " exists, going to remove? [y]")
        #     if ans == "y":
        #         os.remove(demo_path)
        #     else:
        #         return
        date_time = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S-%f")
        demo_path = os.path.join(self.save_path, "demo" + date_time + ".pkl")
        self.robot.go_home()
        self.robot_initial_quat = self.robot.get_ee_pose()[1]
        self.robot.start_cartesian_impedance()
        # self._keyboard_thr.start()
        self._camera_thr.start()
        self._joystick_thr.start()
        # self._command_thr.start()
        # fig, ax = plt.subplots(1, 1)
        
        eef_quat = (rotation.from_quat(torch.Tensor([0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)])) * \
            rotation.from_quat(torch.Tensor([1.0, 0, 0, 0]))).as_quat()
        print("EEF quat", eef_quat)
        old_robot_state = (self.robot.get_joint_positions(), self.gripper.get_state().width)
        file_count = 0
        raw_data_count = 0
        while True:
            if self.stopped:
                break
            if not self.start_or_pause:
                time.sleep(0.1)
                continue
            record_obj = dict()
            # self.read_image_lock = True
            try:
                camera_image, stamp = self._image_buffer.pop()
            except:
            #     self.read_image_lock = False
                continue
            # self.read_image_lock = False
            robot_state = self.robot.get_robot_state()
            gripper_state = self.gripper.get_state()
            command = self._command_queue.get()
            if command["type"] == "gripper_toggle":
                if not gripper_state.is_moving:
                    if gripper_state.is_grasped or gripper_state.width < 1e-3:
                        self.gripper.goto(0.08, 0.1, 1)
                        record_obj["desired_gripper"] = "open"
                    else:
                        self.gripper.grasp(0.1, 1)
                        record_obj["desired_gripper"] = "close"
                    # wait for finger
                    time.sleep(0.5)
                    while self.gripper.get_state().is_moving:
                        time.sleep(0.5)
                else:
                    # skip this command
                    self._command_queue.task_done()
                    continue
            elif command["type"] == "eef":
                eef_pos, _ = self.robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions)) 
                eef_pos = eef_pos + command["delta_pos"]
                try:
                    self.robot.update_desired_ee_pose(eef_pos, eef_quat)
                except grpc.RpcError as e:
                    print(e)
                    self.robot.start_cartesian_impedance()
                    self.robot.update_desired_ee_pose(eef_pos, eef_quat)
            else:
                raise NotImplementedError
            # time.sleep(0.5)
            self._command_queue.task_done()
            raw_data_count += 1
            if (torch.Tensor(robot_state.joint_positions) - old_robot_state[0]).abs().max() > 0.02 \
                or abs(gripper_state.width - old_robot_state[1]) > 0.01:
                if self.save_file:
                    with open(demo_path, "ab") as f:
                        record_obj.update(dict(
                            robot_state=robot_state, gripper_state=gripper_state, 
                            image=camera_image.astype(np.uint8) if camera_image.shape[2] == 3 else camera_image.astype(np.float32), 
                            image_timestamp=stamp,
                            # desired_eef_pos=eef_pos, desired_eef_quat=eef_quat, # 
                        ))
                        pickle.dump(record_obj, f)
                        file_count += 1
                        print("robot", robot_state.timestamp)
                old_robot_state = (torch.Tensor(robot_state.joint_positions), gripper_state.width)
        print("Demo count", file_count, "Raw data count", raw_data_count)
        
    def _camera_listener(self):
        old_timestampe = None
        while True:
            if not self.start_or_pause:
                time.sleep(0.1)
                continue
            image, timestamp = self.camera.read_once()
            if timestamp != old_timestampe:
                self._image_buffer.append((image, timestamp))
                cv2.imshow(
                    f"Video", cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                cv2.waitKey(25)
                old_timestampe = timestamp
    
    def _joystick_listener(self):
        dx, dy, dz = 0., 0., 0.
        scale = 0.01
        while True:
            events = pygame.event.get([pygame.JOYAXISMOTION, pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION])
            gripper_toggle = False
            for event in events:
                if event.type == pygame.JOYAXISMOTION:
                    cur_value = event.value
                    if event.axis == 3:
                        dy = scale * cur_value
                        # if event.value > 0:
                        #     dy = max(dy, 0.01 * cur_value)
                        # else:
                        #     dy = min(dy, 0.01 * cur_value)
                    elif event.axis == 4:
                        dx = scale * cur_value
                        # if event.value > 0:
                        #     dx = max(dx, 0.01 * cur_value)
                        # else:
                        #     dx = min(dx, 0.01 * cur_value)
                    elif event.axis == 1:
                        dz = -scale * cur_value
                        # if event.value > 0:
                        #     dz = min(dz, -0.01 * cur_value)
                        # else:
                        #     dz = max(dz, -0.01 * cur_value)
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 4:
                        # if not gripper_state.is_moving:
                        gripper_toggle = True
                    elif event.button == 0: #A
                        self.start_or_pause = not self.start_or_pause
                        print("Current status:", "start" if self.start_or_pause else "pause")
                    elif event.button == 1: #B
                        self.stopped = True
                        print("Stopped")
            if gripper_toggle:
                try:
                    self._command_queue.put({"type": "gripper_toggle"}, block=False)
                except queue.Full:
                    pass
                # self._command_queue.join()
            elif abs(dx) > scale * 0.01 or abs(dy) > scale * 0.01 or abs(dz) > scale * 0.01:
                delta_pos = torch.Tensor([dx, dy, dz])
                try:
                    self._command_queue.put({"type": "eef", "delta_pos": delta_pos}, block=False)
                except queue.Full:
                    pass
                # self._command_queue.join()

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
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--ip", default="localhost", type=str)
    arg_parser.add_argument("--save_path", default=None, type=str)
    args = arg_parser.parse_args()
    demo_controller = DemoCollector(ip_address=args.ip, save_file=True, save_path=args.save_path)
    demo_controller.run()
