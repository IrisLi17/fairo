import argparse
from polymetis import RobotInterface, GripperInterface, CameraInterface
from bc.bc_network import FCNetwork, EncoderFCNetwork, DiscreteNetwork
from r3m import load_r3m
import torchcontrol.transform.rotation as rotation

import numpy as np
from PIL import Image
import threading
import time
import torch
import torchvision.transforms as T


class NeuralController:
    def __init__(self, ip_address="localhost", model_path=None) -> None:
        self.robot_interface = RobotInterface(ip_address=ip_address)
        self.camera_interface = CameraInterface(ip_address=ip_address)
        self.gripper_interface = GripperInterface(ip_address=ip_address)
        self.camera_thr = threading.Thread(target=self.camera_listener, daemon=True)
        self.current_image: np.ndarray = None
        self.read_image_lock = False
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = load_r3m("resnet50")
        encoder.to(device)
        encoder.eval()
        # controller = FCNetwork(obs_dim=2048 + 11, act_dim=4, hidden_sizes=(256, 256))
        controller = DiscreteNetwork(obs_dim=2048+11, act_dim=(3, 3, 3, 3), hidden_sizes=(256, 256))
        controller.to(device)
        save_obj = torch.load(model_path, map_location=device)
        controller.load_state_dict(save_obj["model"])
        controller.eval()
        self.eef_scale = save_obj["eef_scale"]
        def _image_transform(raw_image: np.ndarray):
            image = Image.fromarray(raw_image.astype(np.uint8))
            fn = T.Compose([T.CenterCrop(224), T.ToTensor()])
            image = fn(image)
            image = image.reshape((-1, 3, 224, 224)).to(self.device) * 255.0
            return image
        self.deploy_policy = EncoderFCNetwork(encoder, controller, _image_transform)
        self.record_image = False  # will consume much memory if set to true
        self.replay_buffer = []

    def loop(self):
        # TODO: multiple trajectories with human reset
        self.camera_thr.start()
        self.robot_interface.go_home()
        # TODO: sometimes communication errors occur
        self.robot_interface.start_cartesian_impedance()
        eef_quat = (rotation.from_quat(torch.Tensor([0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)])) * \
            rotation.from_quat(torch.Tensor([1.0, 0, 0, 0]))).as_quat()
        for i in range(200):
            self.read_image_lock = True
            raw_image = self.current_image.copy()
            self.read_image_lock = False
            robot_state = self.robot_interface.get_robot_state()
            gripper_state = self.gripper_interface.get_state()
            joint_positions = torch.Tensor(robot_state.joint_positions).to(self.device)
            eef_pos, _ = self.robot_interface.robot_model.forward_kinematics(joint_positions)
            gripper_width = torch.Tensor([gripper_state.width]).to(self.device)
            propriocep = torch.concat([joint_positions, eef_pos, gripper_width])
            with torch.no_grad():
                action, rl_obs, processed_image = self.deploy_policy(raw_image, propriocep.unsqueeze(dim=0))
                action = action.squeeze(dim=0)
            print(i, "action", action)
            desired_eef_pos = eef_pos + action[:3] * self.eef_scale
            # Safety clip
            desired_eef_pos = torch.clamp(
                desired_eef_pos, 
                torch.Tensor([0.1, -0.35, 0.1]).to(self.device), 
                torch.Tensor([0.7, 0.35, 0.7]).to(self.device)
            ).cpu()
            print("desired eef pos", desired_eef_pos, "eef quat", eef_quat)
            self.robot_interface.update_desired_ee_pose(desired_eef_pos, eef_quat)
            if abs(action[3].item()) > 0.5:
                if action[3].item() > 0:
                    self.gripper_interface.goto(0.08, 0.1, 1)
                else:
                    self.gripper_interface.grasp(0.1, 1)
                time.sleep(1)
            transition = dict(obs=rl_obs.squeeze(dim=0), action=action)
            if self.record_image:
                transition["image"] = processed_image.squeeze(dim=0).cpu().numpy().astype(np.uint8)
            self.replay_buffer.append(transition)
            time.sleep(0.5)

    def camera_listener(self):
        while True:
            if not self.read_image_lock:
                image, stamp = self.camera_interface.read_once()
                self.current_image = image
            else:
                time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    neural_controller = NeuralController(ip_address="101.6.103.171", model_path=args.model_path)
    neural_controller.loop()
