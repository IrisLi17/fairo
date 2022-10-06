from collections import deque
from polymetis import RobotInterface, GripperInterface, CameraInterface
import torch
import time
import threading


class TransporterController:
    def __init__(
        self, ip_address="localhost", 
        inference_model: torch.nn.Module = None
    ) -> None:
        self.robot = RobotInterface(
            ip_address=ip_address,
        )
        self.gripper = GripperInterface(
            ip_address=ip_address
        )
        self.camera = CameraInterface(
            ip_address=ip_address
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_model = inference_model
        self.inference_model.to(self.device)
        self.language = ""
        self.camera_thr = threading.Thread(target=self.camera_listener, daemon=True)
        self.camera_image_buffer = deque(maxlen=5)
        self.camera_read_lock = False
    
    def run(self):
        self.robot.set_home_pose(torch.Tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]))
        self.robot.go_home()
        _, eef_quat = self.robot.get_ee_pose()
        eef_quat = eef_quat.to(self.device)
        self.language = input("Type language instruction:")
        step = 0
        while True:
            print(f"step {step}")
            # Time average of images
            self.camera_read_lock = True
            image = torch.stack(
                [torch.from_numpy(self.camera_image_buffer[i]) for i in range(len(self.camera_image_buffer))], dim=0
            ).mean(dim=0)
            self.camera_read_lock = False
            # Query cliport
            pick_point, place_point = self.inference_model(image, self.language)
            pick_depth = image[pick_point[0], pick_point[1], -1]
            cam_pick_3d = self.camera.deproject_pixel_to_point(pick_point, pick_depth)
            pick_3d = O_T_cam @ cam_pick_3d
            place_depth = image[place_point[0], place_point[1], -1]
            cam_place_3d = self.camera.deproject_pixel_to_point(place_point, place_depth)
            place_3d = O_T_cam @ cam_place_3d
            approach_pos = pick_3d + torch.Tensor([0.0, 0.0, 0.05], device=self.device)
            self.robot.move_to_ee_pose(approach_pos, eef_quat)
            self.robot.move_to_ee_pose(pick_3d, eef_quat)
            self.gripper.grasp(0.1, 1)
            while True:
                time.sleep(0.5)
                gripper_state = self.gripper.get_state()
                if gripper_state.is_grasped and (not gripper_state.is_moving):
                    break
            self.robot.move_to_ee_pose(approach_pos, eef_quat)
            self.robot.move_to_ee_pose(place_3d, eef_quat)
            self.gripper.goto(0.08, 0.1, 1)
            while True:
                time.sleep(0.5)
                gripper_state = self.gripper.get_state()
                if (not gripper_state.is_grasped) and (not gripper_state.is_moving):
                    break
    
    def camera_listener(self):
        old_timestamp = None
        while True:
            image, timestamp = self.camera.read_once()
            if (not self.camera_read_lock) and timestamp != old_timestamp:
                self.camera_image_buffer.append(image)
                old_timestamp = timestamp
