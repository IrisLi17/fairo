import argparse
import os
from polymetis import CameraInterface, RobotInterface
import cv2
import numpy as np
import torch
import torchcontrol.transform.rotation as rotation
import matplotlib.pyplot as plt
import pickle


def main(args):
    robot_interface = RobotInterface(ip_address=args.ip)
    camera_interface = CameraInterface(ip_address=args.ip)
    camera_intrinsic = camera_interface.get_intrinsic()
    camera_matrix = np.array(
        [
            [camera_intrinsic["fx"], 0., camera_intrinsic["ppx"]],
            [0., camera_intrinsic["fy"], camera_intrinsic["ppy"]],
            [0., 0., 1.],
        ], dtype=np.float32
    )
    dist_coeffs = camera_intrinsic["coeffs"]
    assert os.path.exists(args.calibration_file)
    if args.calibration_file.endswith(".pkl"):
        with open(args.calibration_file, "rb") as f:
            calibration_result = pickle.load(f)
            base_T_cam = calibration_result["base_T_cam"]  
    elif args.calibration_file.endswith(".yml") or args.calibration_file.endswith(".yaml"):
        import yaml
        with open(args.calibration_file, "r") as f:
            calibration_result = yaml.safe_load(f)
        _trans: dict = calibration_result["transformation"]
        base_T_cam = np.eye(4)
        base_T_cam[:3, 3] = np.array([_trans["x"], _trans["y"], _trans["z"]])
        base_T_cam[:3, :3] = rotation.from_quat(
            torch.Tensor([_trans["qx"], _trans["qy"], _trans["qz"], _trans["qw"]])
        ).as_matrix().numpy()
    else:
        raise NotImplementedError
    fig, ax = plt.subplots(1, 1)
    while True:
        image, stamp = camera_interface.read_once()
        joint_positions = robot_interface.get_joint_positions()
        link_pos, link_quat = robot_interface.robot_model.forward_kinematics(joint_positions, "panda_link8")
        base_T_flange = np.concatenate(
            [np.concatenate(
                [rotation.from_quat(link_quat).as_matrix().numpy(), link_pos.reshape(3, 1).numpy()], axis=-1), 
            np.array([[0., 0., 0., 1.]])], axis=0
        )
        r = 0.03  # TODO
        flange_local_points = np.array(
            [[r * np.cos(i * 2 * np.pi / 24), r * np.sin(i * 2 * np.pi / 24)] for i in range(24)]
        )
        cam_flange_points = np.linalg.inv(base_T_cam) @ base_T_flange @ np.transpose(
            np.concatenate(
                [flange_local_points, np.zeros((flange_local_points.shape[0], 1)), np.ones((flange_local_points.shape[0], 1))], axis=-1
            )
        )
        cam_flange_points = np.transpose(cam_flange_points[:3])
        image_points, _ = cv2.projectPoints(cam_flange_points, np.array([0., 0., 0.]), np.array([0., 0., 0.]), camera_matrix, dist_coeffs)
        image_points = np.squeeze(image_points, axis=1)
        # Overlay to original image
        image = image.astype(np.uint8)
        for i in range(len(image_points)):
            x, y = int(image_points[i][0]), int(image_points[i][1])
            image[min(max(y - 1, 0), image.shape[0]): min(max(y + 2, 0), image.shape[0]), 
                  min(max(x - 1, 0), image.shape[1]): min(max(x + 2, 0), image.shape[1])] = np.array([255, 0, 0], dtype=np.uint8)
        ax.cla()
        ax.imshow(image)
        plt.pause(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", default="localhost", type=str)
    parser.add_argument("--calibration_file", default=None, type=str)
    args = parser.parse_args()
    main(args)
