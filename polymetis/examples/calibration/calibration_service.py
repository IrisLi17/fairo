from collections import deque
import time
import threading
from typing import Literal, Tuple
import cv2  # opencv-contrib-python
import numpy as np
from polymetis import CameraInterface
import matplotlib.pyplot as plt
import termios, tty, sys, os
from copy import deepcopy


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
            (ret, charuco_corners, charuco_ids) = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, self.board, cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs)
        else:
            (ret, charuco_corners, charuco_ids) = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, self.board)
        if charuco_ids is None or len(charuco_ids) == 0:
            raise RuntimeError("Cannot detect checker board corners")
        charuco_corners = np.squeeze(charuco_corners, axis=1)
        charuco_ids = np.squeeze(charuco_ids, axis=1)
        return corners, ids, charuco_corners, charuco_ids
    
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
    
    def draw_markers(self, image, marker_corners, charuco_corners=None):
        image = image.copy().astype(np.uint8)
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
    def __init__(self, ip_address) -> None:
        self.camera_interface = CameraInterface(ip_address=ip_address)
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
        self.capture_image_lock = False
        self.current_image_and_feature = None
        self.captured_data = deque(maxlen=20)
        self.merged_feature_image = np.zeros_like(image, dtype=np.uint8)
        self.calibration_result = None
        self.keyboard_monitor_thr = threading.Thread(target=self.keyboard_monitor, daemon=True)
    
    def run(self):
        self.keyboard_monitor_thr.start()
        self.visualize_loop()
    
    def keyboard_monitor(self):
        while True:
            pressed_key = self._getkey()
            if pressed_key == "c":
                self.capture_image_lock = True
                self.captured_data.append(deepcopy(self.current_image_and_feature))
                # update merged features
                for i in range(len(self.current_image_and_feature[1])):
                    point = self.current_image_and_feature[1][i]
                    self.merged_feature_image[
                        max(int(point[1]) - 1, 0): min(int(point[1]) + 2, self.merged_feature_image.shape[0]), 
                        max(int(point[0]) - 1, 0): min(int(point[0]) + 2, self.merged_feature_image.shape[1])
                    ] = np.array([255, 0, 0], dtype=np.uint8)
                print(f"Capture {len(self.captured_data)}")
                self.capture_image_lock = False
            elif pressed_key == "r":
                if len(self.captured_data) < 3:
                    print("Insufficient captures, please collect more")
                    continue
                self.capture_image_lock = True
                all_charuco_corners = [self.captured_data[i][1] for i in range(len(self.captured_data))]
                all_charuco_ids = [self.captured_data[i][2] for i in range(len(self.captured_data))]
                reproj_error, cam_matrix, dist_coeffs = self.calibration_backend.calibrate_intrinsic(all_charuco_corners, all_charuco_ids)
                self.calibration_result = {"cam_matrix": cam_matrix, "dist_coeffs": dist_coeffs, "reproj_error": reproj_error}
                print(f"reproj error: {reproj_error}\ncamera matrix: {cam_matrix}\ndist coeffs: {dist_coeffs}")
                self.capture_image_lock = False
            elif pressed_key == "s":
                import pickle
                save_path = "calib_intrinsic.pkl"
                with open(save_path, "wb") as f:
                    pickle.dump(self.calibration_result, f)
                print(f"Calibration result saved to {save_path}")

    def visualize_loop(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        while True:
            if not self.capture_image_lock:
                image, stamp = self.camera_interface.read_once()
                marker_corners, marker_ids, charuco_corners, charuco_ids = self.calibration_backend.detect(image)
                self.current_image_and_feature = (image, charuco_corners, charuco_ids)
                new_image = self.calibration_backend.draw_markers(image, marker_corners, charuco_corners)
                ax[0].cla()
                ax[0].imshow(new_image)
                ax[0].set_title(stamp)
                ax[1].cla()
                ax[1].imshow(self.merged_feature_image)
                plt.pause(0.1)
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
        

if __name__ == "__main__":
    calibrator = CalibrationService("localhost")
    calibrator.run()
