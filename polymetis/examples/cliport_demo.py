import argparse
import queue
from polymetis import RobotInterface, CameraInterface, GripperInterface
import torchcontrol.transform.rotation as rotation
from transporter_deploy import DummyDataset, CameraConfig
import numpy as np
from datetime import datetime
import os
import cv2
import pickle
import pygame
import threading
import time
import torch
import yaml


class DemoCollector:
    def __init__(self, ip_address: str, folder_name: str, calibration_file: str):
        self.robot_interface = RobotInterface(ip_address=ip_address)
        self.camera_interface = CameraInterface(ip_address=ip_address)
        self.gripper_interface = GripperInterface(ip_address=ip_address)

        self.folder_name = folder_name

        self.demo_path = None
        # self.trigger_obs = False
        # self.trigger_save = False
        self.save_obj = {"obs": [], "action": [], "intrinsics": None}
        # self.command_queue = queue.Queue(maxsize=1)

        self.clicked_positions = []
        self.dummy_dataset = DummyDataset()
        intrinsic_dict = self.camera_interface.get_intrinsic()
        intrinsic_matrix = np.array([
            [intrinsic_dict["fx"], 0, intrinsic_dict["ppx"]],
            [0, intrinsic_dict["fy"], intrinsic_dict["ppy"]],
            [0, 0, 1]
        ])
        if calibration_file.endswith(".yml") or calibration_file.endswith(".yaml"):
            with open(calibration_file, "r") as f:
                calibration_result = yaml.safe_load(f)
            _trans: dict = calibration_result["transformation"]
            base_T_cam = np.eye(4)
            base_T_cam[:3, 3] = np.array([_trans["x"], _trans["y"], _trans["z"]])
            base_T_cam[:3, :3] = rotation.from_quat(
                torch.Tensor([_trans["qx"], _trans["qy"], _trans["qz"], _trans["qw"]])
            ).as_matrix().numpy()
        else:
            with open(calibration_file, "rb") as f:
                calibration_result = pickle.load(f)
            base_T_cam = calibration_result["base_T_cam"]
        self.camera_config = CameraConfig()
        self.camera_config.intrinsic = intrinsic_matrix
        self.camera_config.base_T_cam = base_T_cam
        
        # pygame.init()
        # self.joystick = pygame.joystick.Joystick(0)
        # self.joystick.init()
        # self.joystick_thr = threading.Thread(target=self.joystick_listener, daemon=True)
        # self.control_thr = threading.Thread(target=self.control_callback, daemon=True)
        # self.joystick_thr.start()
        # self.control_thr.start()
    
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

    def loop(self):
        print("What will happen with this tool: you will first need to type the language goal, then press enter. \
               The camera will then capture the scene. Wait until a window named processed poped out. \
               Click the pixel to pick then the pixel to place on the processed image. \
               The robot will start moving after receiving the two clicks. \
               The scripts will automatically dump the demonstration to the folder name you specified and terminates.")
        self.robot_interface.set_home_pose(torch.Tensor([0., -0.785, 0., -2.356, 0., 1.571, 0.785]))
        self.robot_interface.go_home()
        initial_pos, quat = self.robot_interface.get_ee_pose()
        print("initial pos", initial_pos, "quat", quat)

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print("catch mouse", (y, x))
                self.clicked_positions.append((y, x))

        while True:
            if self.demo_path is None:
                self.demo_path = os.path.join(self.folder_name, "demo" + datetime.now().strftime("-%Y-%m-%d-%H-%M-%S-%f") + ".pkl")
                intrinsics = self.camera_interface.get_intrinsic()
                self.save_obj["intrinsics"] = intrinsics
                lang_goal = input("Enter the language goal:")
                self.save_obj["lang_goal"] = lang_goal
            if True:
                rgb_image, depth_image = self.read_images()
                self.save_obj["obs"].append({"color": rgb_image.astype(np.uint8), "depth": depth_image.astype(np.float32)})
                print("Get image")
                processd_image = self.dummy_dataset.get_image(
                    np.concatenate([rgb_image, np.expand_dims(depth_image, axis=-1)], axis=-1), self.camera_config)
                cmap = processd_image[..., :3].astype(np.uint8)
                hmap = processd_image[..., 3]
                cv2.imshow("processed", cv2.cvtColor(cmap, cv2.COLOR_RGB2BGR))
                cv2.setMouseCallback("processed", mouse_cb)
                cv2.waitKey(0)

                while len(self.clicked_positions) < 2:
                    time.sleep(1)
                print("clicked position", self.clicked_positions)
                p0 = pix_to_xyz(self.clicked_positions[0], hmap, self.dummy_dataset.bounds, self.dummy_dataset.pix_size)
                p1 = pix_to_xyz(self.clicked_positions[1], hmap, self.dummy_dataset.bounds, self.dummy_dataset.pix_size)
                p0 = torch.Tensor(p0)
                p1 = torch.Tensor(p1)
                print("converted xyz", p0, p1)
                # convert to link8 position
                p0[2] += 0.1034
                p1[2] += 0.1034
                # Manual offset if you cannot get better calibration. 
                # p0[1] += 0.05
                # p1[1] += 0.05
                approach_pos = p0 + torch.Tensor([0, 0, 0.1])
                # disable for debugging
                self.robot_interface.move_to_ee_pose(approach_pos, quat)
                print("Desired ee pose", p0, quat)
                self.robot_interface.move_to_ee_pose(p0, quat, Kx=1.2 * self.robot_interface.Kx_default)
                ee_pose = self.robot_interface.get_ee_pose()
                print("Resulting ee pose", ee_pose)
                self.save_obj["action"].append({"p0": ee_pose})
                # pick
                self.gripper_interface.grasp(speed=0.1, force=2)
                time.sleep(1)
                while self.gripper_interface.get_state().is_moving:
                    time.sleep(0.1)
                self.robot_interface.move_to_ee_pose(approach_pos, quat)
                place_approach_pos = p1 + torch.Tensor([0, 0, 0.1])
                self.robot_interface.move_to_ee_pose(place_approach_pos, quat, op_space_interp=False)
                self.robot_interface.move_to_ee_pose(p1, quat, Kx=1.2 * self.robot_interface.Kx_default)
                ee_pose = self.robot_interface.get_ee_pose()
                self.save_obj["action"][-1]["p1"] = ee_pose
                # release
                self.gripper_interface.goto(width=0.08, speed=0.1, force=1)
                time.sleep(1)
                while self.gripper_interface.get_state().is_moving:
                    time.sleep(1)
                self.robot_interface.move_to_ee_pose(place_approach_pos, quat)
                # reset
                self.robot_interface.go_home()
                image, stamp = self.camera_interface.read_once()
                self.save_obj["final_obs"] = image.copy()
                
                print("Saving")
                with open(self.demo_path, "wb") as f:
                    pickle.dump(self.save_obj, f)
                print("Demo saved to", self.demo_path)
                break
                
            elif len(self.save_obj["obs"]) == 0:
                time.sleep(0.1)
                continue
            # if self.trigger_save:
                
            time.sleep(0.1) 
            

def pix_to_xyz(pixel, height, bounds, pixel_size, skip_height=False):
    """Convert from pixel location on heightmap to 3D position."""
    u, v = pixel
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 0] + u * pixel_size
    if not skip_height:
        z = bounds[2, 0] + height[u, v]
    else:
        z = 0.0
    return (x, y, z)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--ip", default="localhost", type=str)
    arg_parser.add_argument("--folder_name", default=None, type=str)
    arg_parser.add_argument("--calibration_file", default=None, type=str)
    args = arg_parser.parse_args()
    assert args.folder_name is not None
    if not os.path.exists(args.folder_name):
        os.makedirs(args.folder_name)
    collector = DemoCollector(args.ip, args.folder_name, args.calibration_file)
    collector.loop()
