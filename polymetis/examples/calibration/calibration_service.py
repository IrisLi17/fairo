import argparse
from collections import deque
from enum import Enum
import time
import threading
from typing import List, Literal, Tuple
import cv2  # opencv-contrib-python
import numpy as np
from polymetis import CameraInterface, RobotInterface
import matplotlib.pyplot as plt
import termios, tty, sys, os
from copy import deepcopy


class BoardTypes(Enum):
    CHARUCO = 1
    CIRCLE = 2


class CalibrationBackend:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_width: int, image_height: int) -> None:
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.square_x = 8
        self.square_y = 6
        self.checker_size = 0.02
        self.marker_size = 0.016
        self.board = cv2.aruco.CharucoBoard_create(
            self.square_x, self.square_y, self.checker_size, self.marker_size, self.aruco_dict
        )
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.image_width = image_width
        self.image_height = image_height

    def detect(self, image: np.ndarray, use_intrinsic=False):
        image = image.copy().astype(np.uint8)
        aruco_parameters = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=aruco_parameters)
        if ids is None or len(ids) == 0:
            raise RuntimeError("Cannot detect charuco markers")
        corners = np.array(corners).squeeze(axis=1)
        ids = np.array(ids).flatten()
        if use_intrinsic:
            print("Warning: use_intrinsic may leads to weird results")
            (ret, charuco_corners, charuco_ids) = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, self.board, cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs)
        else:
            (ret, charuco_corners, charuco_ids) = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, self.board)
        if charuco_ids is None or len(charuco_ids) == 0:
            raise RuntimeError("Cannot detect checker board corners")
        is_valid, board_rvec, board_tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board, self.camera_matrix, self.dist_coeffs, None, None
        )
        if not is_valid:
            board_rvec = None
            board_tvec = None
        charuco_corners = np.squeeze(charuco_corners, axis=1)
        charuco_ids = np.squeeze(charuco_ids, axis=1)
        return corners, ids, charuco_corners, charuco_ids, board_rvec, board_tvec
    
    def calibrate_intrinsic(self, list_charuco_corners, list_charuco_ids):
        # marker_corners, marker_ids, charuco_corners, charuco_ids = self.detect(image, use_intrinsic=False)
        assert len(list_charuco_corners) == len(list_charuco_ids)
        list_object_points = []
        for i in range(len(list_charuco_corners)):
            object_xy = [
                [[self.checker_size * i, -self.checker_size * j] for i in range(self.square_x - 1)] 
                for j in range(self.square_y - 1)
            ]
            object_xy = np.array(object_xy).reshape(-1, 2)[list_charuco_ids[i]].astype(np.float32)
            object_points = np.concatenate([object_xy, np.zeros((object_xy.shape[0], 1), dtype=np.float32)], axis=-1)
            list_object_points.append(object_points)
        reproj_error, new_cam_matrix, new_dist_coeffs, rvec, tvec, std_intrisic, std_extrinsic, per_view_error = \
            cv2.calibrateCameraExtended(
                list_object_points, list_charuco_corners, np.array([self.image_width, self.image_height]), None, None
            )
        return reproj_error, new_cam_matrix, new_dist_coeffs
    
    # TODO: try to measure error
    def calibrate_handeye(
        self, list_ee_pose: List[np.ndarray], 
        list_r_board: List[np.ndarray], list_t_board: List[np.ndarray], eye_on_hand: bool
    ):
        # list_ee_pose: list of 4X4 ee pose
        # list_r_board: list of 3X1 rotation vector
        # list_t_board: list of 3X1 translation
        assert len(list_ee_pose) == len(list_r_board) and len(list_r_board) == len(list_t_board)
        if eye_on_hand:
            list_r_tcp = [pose[:3, :3] for pose in list_ee_pose]
            list_t_tcp = [pose[:3, 3:] for pose in list_ee_pose]
            r_gripper_cam, t_gripper_cam = cv2.calibrateHandEye(list_r_tcp, list_t_tcp, list_r_board, list_t_board)
            gripper_T_cam = np.concatenate([np.concatenate([r_gripper_cam, t_gripper_cam], axis=-1), np.array([[0., 0., 0., 1.]])], axis=0)
            # base_T_board = base_T_eef @ gripper_T_cam @ cam_T_board
            base_T_board = []
            for i in range(len(list_r_tcp)):
                cam_T_board = np.eye(4)
                cam_T_board[:3, :3] = cv2.Rodrigues(list_r_board[i])[0]
                cam_T_board[:3, 3:] = list_t_board[i]
                base_T_board.append(list_ee_pose[i] @ gripper_T_cam @ cam_T_board)
            base_rvec_board = np.array([cv2.Rodrigues(mat[:3, :3])[0] for mat in base_T_board]).squeeze(axis=-1)
            base_tvec_board = np.array([mat[:3, 3] for mat in base_T_board])
            debug = {"base_rvec_board": base_rvec_board, "base_tvec_board": base_tvec_board}
            return gripper_T_cam, debug
        else:
            # eye on base
            inv_list_ee_pose = [np.linalg.inv(pose) for pose in list_ee_pose]
            inv_list_r_tcp = [pose[:3, :3] for pose in inv_list_ee_pose]
            inv_list_t_tcp = [pose[:3, 3:] for pose in inv_list_ee_pose]
            r_base_cam, t_base_cam = cv2.calibrateHandEye(inv_list_r_tcp, inv_list_t_tcp, list_r_board, list_t_board)
            base_T_cam = np.concatenate([np.concatenate([r_base_cam, t_base_cam], axis=-1), np.array([[0., 0., 0., 1.]])], axis=0)
            # eef_T_board = (ee_pose).inv @ base_T_cam @ cam_T_board
            cam_T_board = [np.eye(4) for _ in range(len(list_r_board))]
            eef_T_board = []
            for i in range(len(list_r_board)):
                cam_T_board[i][:3, :3] = cv2.Rodrigues(list_r_board[i])[0]
                cam_T_board[i][:3, 3:] = list_t_board[i]
                eef_T_board.append(inv_list_ee_pose[i] @ base_T_cam @ cam_T_board[i])
            eef_rvec_board = np.array([cv2.Rodrigues(mat[:3, :3])[0] for mat in eef_T_board]).squeeze(axis=-1)
            eef_tvec_board = np.array([mat[:3, 3] for mat in eef_T_board])
            debug = {"eef_rvec_board": eef_rvec_board, "eef_tvec_board": eef_tvec_board}
            return base_T_cam, debug

    def draw_markers(self, image, marker_corners=None, charuco_corners=None):
        image = image.copy().astype(np.uint8)
        if marker_corners is not None:
            for i in range(len(marker_corners)):
                xys = marker_corners[i]
                for j in range(xys.shape[0]):
                    image[
                        max(int(xys[j][1]) - 1, 0): min(int(xys[j][1]) + 2, image.shape[0]), 
                        max(int(xys[j][0]) - 1, 0): min(int(xys[j][0]) + 2, image.shape[1])
                    ] = np.array([255, 0, 0], dtype=np.uint8)
        if charuco_corners is not None:
            for i in range(len(charuco_corners)):
                image[int(charuco_corners[i][1]), int(charuco_corners[i][0])] = np.array([0, 255, 0], dtype=np.uint8)
        return image

class CalibrationService:
    KEY_MAPPINGS = {
        32: "space",
        27: "esc"
    }
    def __init__(self, ip_address, calibration_type="handeye") -> None:
        self.camera_interface = CameraInterface(ip_address=ip_address)
        try:
            self.robot_interface = RobotInterface(ip_address=ip_address)
        except:
            print("Warning: robot not connected")
            assert calibration_type != "handeye"
            self.robot_interface = None
        camera_intrinsic = self.camera_interface.get_intrinsic()
        camera_matrix = np.array(
            [
                [camera_intrinsic["fx"], 0., camera_intrinsic["ppx"]],
                [0., camera_intrinsic["fy"], camera_intrinsic["ppy"]],
                [0., 0., 1.],
            ], dtype=np.float32
        )
        dist_coeffs = camera_intrinsic["coeffs"]
        image, stamp = self.camera_interface.read_once()
        self.calibration_backend = CalibrationBackend(camera_matrix, dist_coeffs, image.shape[1], image.shape[0])
        assert calibration_type in ["intrinsic", "handeye"]
        self.calibration_type = calibration_type
        self.capture_image_lock = False
        self.current_image_and_feature = dict(
            image=None, charuco_corners=None, charuco_ids=None, board_rvec=None, board_tvec=None,
            ee_pose=None,
        )
        self.captured_data = deque(maxlen=20)
        self.merged_feature_image = np.zeros_like(image, dtype=np.uint8)
        self.calibration_result = None
        self.keyboard_monitor_thr = threading.Thread(target=self.keyboard_monitor, daemon=True)
    
    def run(self):
        self.keyboard_monitor_thr.start()
        self.visualize_loop()
    
    def _update_merged_features(self):
        self.merged_feature_image = np.zeros_like(self.merged_feature_image)
        for i in range(len(self.captured_data)):
            for point in self.captured_data[i]["charuco_corners"]:
                self.merged_feature_image[
                    max(int(point[1]) - 1, 0): min(int(point[1]) + 2, self.merged_feature_image.shape[0]), 
                    max(int(point[0]) - 1, 0): min(int(point[0]) + 2, self.merged_feature_image.shape[1])
                ] = np.array([255, 0, 0], dtype=np.uint8)

    def keyboard_monitor(self):
        while True:
            pressed_key = self._getkey()
            if pressed_key == "c":
                self.capture_image_lock = True
                self.captured_data.append(deepcopy(self.current_image_and_feature))
                self._update_merged_features()
                print(f"Capture {len(self.captured_data)}")
                self.capture_image_lock = False
            elif pressed_key == "d":
                self.captured_data.pop()
                self._update_merged_features()
                print(f"Removed the latest capture, remaining capture number {len(self.captured_data)}")
            elif pressed_key == "r":
                if len(self.captured_data) < 3:
                    print("Insufficient captures, please collect more")
                    continue
                self.capture_image_lock = True
                if self.calibration_type == "intrinsic":
                    all_charuco_corners = [self.captured_data[i]["charuco_corners"] for i in range(len(self.captured_data))]
                    all_charuco_ids = [self.captured_data[i]["charuco_ids"] for i in range(len(self.captured_data))]
                    reproj_error, cam_matrix, dist_coeffs = self.calibration_backend.calibrate_intrinsic(all_charuco_corners, all_charuco_ids)
                    self.calibration_result = {"cam_matrix": cam_matrix, "dist_coeffs": dist_coeffs, "reproj_error": reproj_error}
                    print(f"reproj error: {reproj_error}\ncamera matrix: {cam_matrix}\ndist coeffs: {dist_coeffs}")
                elif self.calibration_type == "handeye":
                    all_ee_pose = [self.captured_data[i]["ee_pose"] for i in range(len(self.captured_data))]
                    all_board_rvec = [self.captured_data[i]["board_rvec"] for i in range(len(self.captured_data))]
                    all_board_tvec = [self.captured_data[i]["board_tvec"] for i in range(len(self.captured_data))]
                    base_T_cam, debug_info = self.calibration_backend.calibrate_handeye(
                        all_ee_pose, all_board_rvec, all_board_tvec, eye_on_hand=False
                    )
                    self.calibration_result = {"base_T_cam": base_T_cam, "debug_info": debug_info}
                    print(f"base_T_cam", self.calibration_result)
                self.capture_image_lock = False
            elif pressed_key == "s":
                import pickle
                if self.calibration_type == "intrinsic":
                    save_path = "calib_intrinsic.pkl"
                elif self.calibration_type == "handeye":
                    save_path = "calib_handeye.pkl"
                with open(save_path, "wb") as f:
                    pickle.dump(self.calibration_result, f)
                print(f"Calibration result saved to {save_path}")
                with open("calib_raw_data.pkl", "wb") as f:
                    pickle.dump(self.captured_data, f)
                print(f"Raw data saved to calib_raw_data.pkl")

    def visualize_loop(self):
        # fig, ax = plt.subplots(1, 2, figsize=(18, 8))
        while True:
            if not self.capture_image_lock:
                image, stamp = self.camera_interface.read_once()
                try:
                    marker_corners, marker_ids, charuco_corners, charuco_ids, board_rvec, board_tvec = self.calibration_backend.detect(image)
                    # self.current_image_and_feature = (image, charuco_corners, charuco_ids, board_rvec, board_tvec)
                    self.current_image_and_feature.update(
                        dict(
                            image=image.astype(np.uint8), charuco_corners=charuco_corners, charuco_ids=charuco_ids,
                            board_rvec=board_rvec, board_tvec=board_tvec,
                        )  
                    )
                except RuntimeError:
                    marker_corners = charuco_corners = None
                if self.robot_interface is not None:
                    ee_pos, ee_quat = self.robot_interface.get_ee_pose()
                    import torchcontrol.transform.rotation as rotation
                    ee_rot_mat = rotation.from_quat(ee_quat).as_matrix()
                    ee_pose = np.concatenate([np.concatenate([ee_rot_mat.numpy(), ee_pos.numpy().reshape(3, 1)], axis=-1), np.array([[0, 0, 0, 1]])], axis=0)
                    self.current_image_and_feature.update(
                        dict(
                            ee_pose=ee_pose
                        )
                    )
                new_image = self.calibration_backend.draw_markers(image, marker_corners, charuco_corners)
                # ax[0].cla()
                # ax[0].imshow(new_image)
                # ax[0].set_title(stamp)
                # ax[1].cla()
                # ax[1].imshow(self.merged_feature_image)
                # plt.pause(0.1)
                cv2.imshow("Calibration", cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
                cv2.imshow("Features", cv2.cvtColor(self.merged_feature_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(70)
            else:
                time.sleep(0.5)
    
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
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
    def __del__(self):
        # TODO: how to handle when the program exited with error
        os.system("stty sane")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", default="localhost", type=str)
    parser.add_argument("--calibration_type", default="handeye", choices=["handeye", "intrinsic"])
    args = parser.parse_args()
    calibrator = CalibrationService(args.ip, calibration_type=args.calibration_type)
    # calibrator = CalibrationService("101.6.103.171", calibration_type="handeye")
    calibrator.run()
