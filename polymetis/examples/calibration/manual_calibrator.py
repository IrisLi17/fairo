import argparse
import threading
from polymetis import RobotInterface, CameraInterface
import numpy as np
import torch
import torchcontrol.transform.rotation as rotation
import cv2
import os, pickle, sys, termios, tty


class ManualCalibrator:
    def __init__(self, ip_address: str, calibration_file: str) -> None:
        self.robot_interface = RobotInterface(ip_address=ip_address)
        self.camera_interface = CameraInterface(ip_address=ip_address)
        camera_intrinsic = self.camera_interface.get_intrinsic()
        self.camera_matrix = np.array(
            [
                [camera_intrinsic["fx"], 0., camera_intrinsic["ppx"]],
                [0., camera_intrinsic["fy"], camera_intrinsic["ppy"]],
                [0., 0., 1.],
            ], dtype=np.float32
        )
        self.dist_coeffs = camera_intrinsic["coeffs"]
        assert os.path.exists(calibration_file)
        if calibration_file.endswith(".pkl"):
            with open(calibration_file, "rb") as f:
                calibration_result = pickle.load(f)
                base_T_cam = calibration_result["base_T_cam"]  
        elif calibration_file.endswith(".yml") or calibration_file.endswith(".yaml"):
            import yaml
            with open(calibration_file, "r") as f:
                calibration_result = yaml.safe_load(f)
            _trans: dict = calibration_result["transformation"]
            base_T_cam = np.eye(4)
            base_T_cam[:3, 3] = np.array([_trans["x"], _trans["y"], _trans["z"]])
            base_T_cam[:3, :3] = rotation.from_quat(
                torch.Tensor([_trans["qx"], _trans["qy"], _trans["qz"], _trans["qw"]])
            ).as_matrix().numpy()
        else:
            raise NotImplementedError
        self.base_T_cam = base_T_cam
        self.euler_angles = rotmat2euler(base_T_cam[:3, :3])
        self.pos_scale = 0.001
        self.rad_scale = np.pi / 180
        self.keyboard_thr = threading.Thread(target=self.update_transform, daemon=True)
        
    def run(self):
        self.keyboard_thr.start()
        self.loop()

    def update_transform(self):
        while True:
            pressed_key = self._getkey()
            if pressed_key == "a":
                self.base_T_cam[0, 3] += self.pos_scale
            elif pressed_key == "d":
                self.base_T_cam[0, 3] -= self.pos_scale
            elif pressed_key == "w":
                self.base_T_cam[1, 3] += self.pos_scale
            elif pressed_key == "s":
                self.base_T_cam[1, 3] -= self.pos_scale
            elif pressed_key == "i":
                self.base_T_cam[2, 3] += self.pos_scale
            elif pressed_key == "k":
                self.base_T_cam[2, 3] -= self.pos_scale
            elif pressed_key == "r":
                self.euler_angles[0] += self.rad_scale
            elif pressed_key == "e":
                self.euler_angles[0] -= self.rad_scale
            elif pressed_key == "p":
                self.euler_angles[1] += self.rad_scale
            elif pressed_key == "o":
                self.euler_angles[1] -= self.rad_scale
            elif pressed_key == "y":
                self.euler_angles[2] += self.rad_scale
            elif pressed_key == "t":
                self.euler_angles[2] -= self.rad_scale
            elif pressed_key == "s":
                print("current calibration", self.base_T_cam)
            self.base_T_cam[:3, :3] = euler2rotmat(self.euler_angles)
    
    def loop(self):
        while True:
            image, stamp = self.camera_interface.read_once()
            joint_positions = self.robot_interface.get_joint_positions()
            link_pos, link_quat = self.robot_interface.robot_model.forward_kinematics(joint_positions, "panda_link8")
            base_T_flange = np.concatenate(
                [np.concatenate(
                    [rotation.from_quat(link_quat).as_matrix().numpy(), link_pos.reshape(3, 1).numpy()], axis=-1), 
                np.array([[0., 0., 0., 1.]])], axis=0
            )
            r = 0.03  # TODO
            flange_local_points = np.array(
                [[r * np.cos(i * 2 * np.pi / 24), r * np.sin(i * 2 * np.pi / 24)] for i in range(24)]
            )
            # x-axis
            axis_local_points = np.array(
                [[i / 10 * r , 0] for i in range(11)]
            )
            flange_local_points = np.concatenate([flange_local_points, axis_local_points], axis=0)
            cam_flange_points = np.linalg.inv(self.base_T_cam) @ base_T_flange @ np.transpose(
                np.concatenate(
                    [flange_local_points, np.zeros((flange_local_points.shape[0], 1)), np.ones((flange_local_points.shape[0], 1))], axis=-1
                )
            )
            cam_flange_points = np.transpose(cam_flange_points[:3])
            image_points, _ = cv2.projectPoints(cam_flange_points, np.array([0., 0., 0.]), np.array([0., 0., 0.]), self.camera_matrix, self.dist_coeffs)
            image_points = np.squeeze(image_points, axis=1)
            # Overlay to original image
            image = image.astype(np.uint8)
            for i in range(len(image_points)):
                x, y = int(image_points[i][0]), int(image_points[i][1])
                color = np.array([255, 0, 0], dtype=np.uint8) if i < len(image_points) - len(axis_local_points) \
                    else np.array([0, 255, 0], dtype=np.uint8)
                image[min(max(y - 1, 0), image.shape[0]): min(max(y + 2, 0), image.shape[0]), 
                    min(max(x - 1, 0), image.shape[1]): min(max(x + 2, 0), image.shape[1])] = color
            cv2.imshow("calibration", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(25)
    
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
                return chr(k)
        except:
            return None
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
    def __del__(self):
        # TODO: how to handle when the program exited with error
        os.system("stty sane")


def rotmat2euler(R: np.ndarray):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


def euler2rotmat(theta: np.array):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
 
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
 
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
 
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--ip", default="localhost", type=str)
    arg_parser.add_argument("--calibration_file", default=None, type=str)
    args = arg_parser.parse_args()
    manual_calibrator = ManualCalibrator(args.ip, args.calibration_file)
    manual_calibrator.run()
