from collections import deque
from polymetis import RobotInterface, GripperInterface, CameraInterface
import torchcontrol.transform.rotation as rotation
import cv2
import numpy as np
import torch
import time
import threading
import os, pickle, yaml


class CameraConfig:
    intrinsic: np.ndarray = np.eye(3)
    base_T_cam: np.ndarray = np.eye(4)

class DummyDataset:
    def __init__(self) -> None:
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0.0, 0.2]])
        self.pix_size = 0.003125
    
    def get_image(self, obs: np.ndarray, cam_config):
        assert obs.shape[-1] == 4  # RGBD
        rgb = obs[..., :3]
        depth = obs[..., 3]
        hmap, cmap = self.reconstruct_image(rgb, depth, cam_config, self.bounds, self.pix_size)
        print(cmap.shape, hmap.shape)
        image = np.concatenate([cmap, np.tile(np.expand_dims(hmap, axis=-1), (1, 1, 3))], axis=-1)
        return image
        
    def get_pointcloud(self, depth_map: np.ndarray, intrinsic: np.ndarray):
        height, width = depth_map.shape
        assert intrinsic.shape[0] == 3 and intrinsic.shape[1] == 3
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        # convert to coordinates
        px = (px - intrinsic[0, 2]) / intrinsic[0, 0] * depth_map
        py = (py - intrinsic[1, 2]) / intrinsic[1, 1] * depth_map
        points = np.stack([px, py, depth_map], axis=-1)
        return points

    def get_heightmap(self, points, colors, bounds, pixel_size):
        """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
    
        Args:
        points: HxWx3 float array of 3D points in world coordinates.
        colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
        bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
            region in 3D space to generate heightmap in world coordinates.
        pixel_size: float defining size of each pixel in meters.
    
        Returns:
        heightmap: HxW float array of height (from lower z-bound) in meters.
        colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
        """
        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
        heightmap = np.zeros((height, width), dtype=np.float32)
        colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
        iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
        iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
        valid = ix & iy & iz
        # Add morphological operation
        kernel = np.ones((5, 5), dtype=np.uint8)
        _dtype = valid.dtype
        valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(_dtype)
        points = points[valid]
        colors = colors[valid]
        
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]
        px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        heightmap[py, px] = points[:, 2] - bounds[2, 0]
        for c in range(colors.shape[-1]):
            colormap[py, px, c] = colors[:, c]
        return heightmap, colormap
    
    def reconstruct_image(self, color_image, depth_map, camera_config: CameraConfig, bounds, pixel_size: float):
        H, W = depth_map.shape
        # first get point cloud in camera frame
        intrinsic = camera_config.intrinsic
        cam2points = self.get_pointcloud(depth_map, intrinsic)
        # then transform to world frame
        base_T_cam = camera_config.base_T_cam
        rotation = base_T_cam[:3, :3]
        translation = base_T_cam[:3, 3:]
        world2points = np.transpose(
            rotation @ np.transpose(cam2points.reshape((H * W, 3))) + translation
        )  # (H * W, 3)
        reshaped_world2points = world2points.reshape((H, W, 3))
        # simulate z-buffering to get top-down height map and corresponding color image
        assert bounds.shape == (3, 2)
        heightmap, colormap = self.get_heightmap(world2points.reshape((H, W, 3)), color_image, bounds, pixel_size)
        return heightmap, colormap
  
'''    
class TransporterController:
    def __init__(
        self, ip_address="localhost", 
        base_T_cam: np.ndarray = None,  # (4X4)
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
        self.base_T_cam = base_T_cam
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
            # TODO: how to make the agent
            # TODO: cliport requires first transform the image to a special top-down format
            assert image.shape[-1] == 4
            info = {'lang_goal': self.language}
            pick_place_result = self.inference_model(image, info)
            pick_xyz, pick_quat = pick_place_result["pose0"]
            
            pick_3d = (self.base_T_cam @ np.reshape(np.concatenate([cam_pick_3d, [1.]]), (4, 1)))[:3, 0]
            # pick_3d = O_T_cam @ cam_pick_3d
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
'''

if __name__ == "__main__":
    camera = CameraInterface(ip_address="101.6.103.171")
    # calibration_file = "/home/yunfei/Downloads/panda_eob_calib_eye_on_base.yaml"
    calibration_file = os.path.join(os.path.dirname(__file__), "calib_handeye.pkl")
    intrinsic_dict = camera.get_intrinsic()
    intrinsic_matrix = np.array([
        [intrinsic_dict["fx"], 0, intrinsic_dict["ppx"]],
        [0, intrinsic_dict["fy"], intrinsic_dict["ppy"]],
        [0, 0, 1]
    ])
    if calibration_file.endswith(".yaml"):
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
            data = pickle.load(f)
        base_T_cam = data["base_T_cam"]
    # o_t_cam = (0.5928584625283155, -0.543854739016389, 0.5912458350222481) # fake
    # o_q_cam = (-0.9062201724320003, -0.16421328818549003, 0.07074543799124425, 0.38313715307191526)
    # rot = rotation.from_quat(
    #     torch.Tensor(o_q_cam)
    # ).as_matrix().numpy()
    depth_image = []
    old_timestamp = None
    count = 0
    while count < 15:
        image_i, stamp = camera.read_once()
        if stamp != old_timestamp:
            rgb_image = image_i[..., :3]
            depth_image.append(image_i[..., 3:])
            count += 1
            old_timestamp = stamp
    depth_image = np.median(np.array(depth_image), axis=0)
    image = np.concatenate([rgb_image, depth_image], axis=-1)
    print("raw image", image.shape, image.dtype)
    import matplotlib.pyplot as plt
    plt.imsave("raw_temp.png", image[..., :3].astype(np.uint8))
    plt.imsave("raw_temp1.png", np.clip(image[..., 3], 0, 1))
    image_size = image.shape[:2]
    cam_config = CameraConfig()
    cam_config.intrinsic = intrinsic_matrix
    cam_config.base_T_cam = base_T_cam 
    dataset = DummyDataset()
    processed_img = dataset.get_image(image, cam_config)
    cmap = processed_img[..., :3]
    hmap = processed_img[..., 3]
    plt.imsave("temp.png", cmap.astype(np.uint8))
    plt.imsave("temp1.png", hmap)