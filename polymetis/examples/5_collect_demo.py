from polymetis import RobotInterface, GripperInterface, CameraInterface
import polymetis_pb2
import keyboard
import queue
import threading
import time
import torch
import pickle


class DemoCollector:
    VALID_KEYS = ["a", "w", "s", "d", "i", "k", "space", "q"]
    def __init__(self, ip_address="localhost") -> None:
        self.robot = RobotInterface(
            ip_address=ip_address
        )
        self.gripper = GripperInterface(
            ip_address=ip_address
        )
        self.camera = CameraInterface(
            ip_address=ip_address
        )
        self._keyboard_queue = queue.Queue(maxsize=1)
        self._robstate_queue = queue.Queue(maxsize=1)
        self._image_queue = queue.Queue(maxsize=1)
        self._keyboard_thr = threading.Thread(
            target=self._keyboard_listener,
            daemon=True,
        )
        self._camera_thr = threading.Thread(
            target=self._camera_listener,
            daemon=True
        )
        self._command_thr = threading.Thread(
            target=self._command_executor,
            daemon=True,
        )
    
    def _keyboard_listener(self):
        print("Listening keyboard events")
        t0 = time.time()
        while True:
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN and event.name in self.VALID_KEYS:
                self._keyboard_queue.put((event.name, time.time() - t0))
            if event.name == "q":
                break
    
    def _camera_listener(self):
        print("Listening camera events")
        old_timestamp = None
        while True:
            image, timestamp = self.camera.read_once()
            if timestamp != old_timestamp:
                self._image_queue.put((image, timestamp))
                old_timestamp = timestamp
    
    def _robstate_listener(self):
        print("Listening robot states")
        while True:
            robot_state: polymetis_pb2.RobotState = self.robot.get_robot_state()
            gripper_state: polymetis_pb2.GripperState = self.gripper.get_state()
            self._robstate_queue.put((robot_state, gripper_state))

    def _command_executor(self):
        self.robot.start_cartesian_impedance()
        while True:
            action, timestamp = self._keyboard_queue.get()
            if self._robstate_queue.qsize() > 0 and self._image_queue.qsize() > 0:
                robot_state, gripper_state = self._robstate_queue.get(block=False)
                camera_image = self._image_queue.get(block=False)
                eef_pos, eef_quat = self.robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions))
                if action == "a":
                    eef_pos = eef_pos + torch.Tensor([0.01, 0.0, 0.0])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif action == "d":
                    eef_pos = eef_pos + torch.Tensor([-0.01, 0.0, 0.0])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif action == "w":
                    eef_pos = eef_pos + torch.Tensor([0.0, 0.01, 0.0])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif action == "s":
                    eef_pos = eef_pos + torch.Tensor([0.0, -0.01, 0.0])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif action == "i":
                    eef_pos = eef_pos + torch.Tensor([0.0, 0.0, 0.01])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif action == "k":
                    eef_pos = eef_pos + torch.Tensor([0.0, 0.0, -0.01])
                    self.robot.update_desired_ee_pose(position=eef_pos, orientation=eef_quat)
                elif action == "space":
                    if not gripper_state.is_moving:
                        if gripper_state.is_grasped:
                            self.gripper.goto(0.08, 0.1, 1)
                        else:
                            self.gripper.grasp(0.1, 1)
                elif action == "q":
                    break
                # Save
                with open("demo.pkl", "ab") as f:
                    record_obj = dict(
                        robot_state=robot_state, gripper_state=gripper_state, image=camera_image, 
                        desired_eef_pos=eef_pos, desired_eef_quat=eef_quat
                    )
                    pickle.dump(record_obj, f)
                    print("Demo saved")
        self.robot.terminate_current_policy()


if __name__ == "__main__":
    demo_controller = DemoCollector(ip_address="101.6.103.171")
    
    