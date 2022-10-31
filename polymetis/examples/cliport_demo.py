import argparse
import queue
from polymetis import RobotInterface, CameraInterface, GripperInterface
import numpy as np
from datetime import datetime
import os
import cv2
import pickle
import pygame
import threading
import time

class DemoCollector:
    def __init__(self, ip_address: str, folder_name: str):
        self.robot_interface = RobotInterface(ip_address=ip_address)
        self.camera_interface = CameraInterface(ip_address=ip_address)
        self.gripper_interface = GripperInterface(ip_address=ip_address)

        self.folder_name = folder_name

        self.demo_path = None
        self.trigger_obs = False
        self.trigger_save = False
        self.save_obj = {"obs": [], "action": [], "intrinsics": None}
        self.command_queue = queue.Queue(maxsize=1)
        
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.joystick_thr = threading.Thread(target=self.joystick_listener, daemon=True)
        self.control_thr = threading.Thread(target=self.control_callback, daemon=True)
        self.joystick_thr.start()
        self.control_thr.start()
    
    def read_images(self):
        depth_image = []
        old_timestamp = None
        count = 0
        while count < 15:
            image_i, stamp = self.camera_interface.read_once()
            if stamp != old_timestamp:
                rgb_image = image_i[..., :3]
                depth_image.append(image_i[..., 3])
                count += 1
                print("In image capture", count)
                old_timestamp = stamp
        depth_image = np.median(np.array(depth_image), axis=0)
        cv2.imshow("color", cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imshow("depth", depth_image)
        cv2.waitKey(1000)
        return rgb_image, depth_image

    def control_callback(self):
        while True:
            self.command_queue.get()
            ee_pose = self.robot_interface.get_ee_pose()
            gripper_state = self.gripper_interface.get_state()
            if gripper_state.is_moving:
                self.command_queue.task_done()
                continue
            if (not gripper_state.is_grasped) and (gripper_state.width >= 1e-3):
                if len(self.save_obj["obs"]) - len(self.save_obj["action"]) < 1:
                    print("Please trigger observation before p0. Action ignored")
                else:
                    self.gripper_interface.grasp(speed=0.1, force=5)
                    self.save_obj["action"].append({"p0": ee_pose})
            else:
                if len(self.save_obj["action"]) == 0 or "p1" in self.save_obj["action"][-1]:
                    print("Not the time to trigger p1. Action ignored")
                else:
                    self.gripper_interface.goto(width=0.08, speed=0.1, force=1)
                    self.save_obj["action"][-1]["p1"] = ee_pose
                    image, stamp = self.camera_interface.read_once()
                    self.save_obj["final_obs"] = image.copy()
            self.command_queue.task_done()

    def loop(self):
        while True:
            if self.demo_path is None:
                self.demo_path = os.path.join(self.folder_name, "demo" + datetime.now().strftime("-%Y-%m-%d-%H-%M-%S-%f") + ".pkl")
                intrinsics = self.camera_interface.get_intrinsic()
                self.save_obj["intrinsics"] = intrinsics
                lang_goal = input("Enter the language goal:")
                self.save_obj["lang_goal"] = lang_goal
            if self.trigger_obs:
                rgb_image, depth_image = self.read_images()
                self.save_obj["obs"].append({"color": rgb_image.astype(np.uint8), "depth": depth_image.astype(np.float32)})
                print("Get image")
                self.trigger_obs = False
            elif len(self.save_obj["obs"]) == 0:
                time.sleep(0.1)
                continue
            if self.trigger_save:
                print("Saving")
                with open(self.demo_path, "wb") as f:
                    pickle.dump(self.save_obj, f)
                print("Demo saved to", self.demo_path)
                break
        time.sleep(0.1) 
            
    def joystick_listener(self):
        while True:
            events = pygame.event.get([pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION])
            for event in events:
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 4:
                        # if not gripper_state.is_moving:
                        try:
                            self.command_queue.put("toggle", block=False)
                        except queue.Full:
                            pass
                    elif event.button == 1: #B
                        self.trigger_save = True
                        print("Save and exit")
                        break
                    elif event.button == 2: # X
                        self.trigger_obs = True
                        print("Trigger obs")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--ip", default="localhost", type=str)
    arg_parser.add_argument("--folder_name", default=None, type=str)
    args = arg_parser.parse_args()
    assert args.folder_name is not None
    if not os.path.exists(args.folder_name):
        os.makedirs(args.folder_name)
    collector = DemoCollector(args.ip, args.folder_name)
    collector.loop()
