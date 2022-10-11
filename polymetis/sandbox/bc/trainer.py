import shutil
from bc.bc_network import FCNetwork, DiscreteNetwork

import numpy as np
from typing import Dict, List, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as T
import torchcontrol as toco
import pickle
from PIL import Image
from collections import deque


class BehaviorCloning:
    def __init__(self, policy: Union[FCNetwork, DiscreteNetwork], encode_fn: nn.Module, device, lr=1e-3) -> None:
        self.policy = policy
        self.encode_fn = encode_fn
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.eef_scale = 0.01
    
    def train(self, demo_paths: List[str], num_epochs=10, batch_size=32):
        expert_demos = self.parse_demo(demo_paths, save_image=True)
        num_samples = len(expert_demos["observation"])
        indices = np.arange(num_samples)
        losses = deque(maxlen=10)
        all_losses = []
        with tqdm(total=num_samples // batch_size) as pbar:
            for e in range(num_epochs):
                np.random.shuffle(indices)
                for mb in range(num_samples // batch_size):
                    rand_idx = indices[mb * batch_size: (mb + 1) * batch_size]
                    # mb_images = expert_demos["image"][rand_idx]
                    # with torch.no_grad():
                    #     mb_embed = self.encode_fn(mb_images)
                    # mb_obs = expert_demos["observation"][rand_idx]
                    # mb_input = torch.concat([mb_embed, mb_obs], dim=-1)
                    mb_input = expert_demos["observation"][rand_idx]
                    mb_action = expert_demos["action"][rand_idx]
                    loss = self.policy.get_loss(mb_input, mb_action)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_description("Epoch %d" % e)
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)
                    losses.append(loss.item())
                    if len(losses) >= 10:
                        all_losses.append(np.mean(losses))
        import matplotlib.pyplot as plt
        plt.plot(all_losses)
        plt.savefig("figure1.png")
        save_obj = dict(
            model=self.policy.state_dict(),
            eef_scale=self.eef_scale
        )
        torch.save(save_obj, "bc_model.pt")
        print("Model saved to", "bc_model.pt")
    
    def evaluate(self, demo_paths: List[str]):
        dataset = self.parse_demo(demo_paths)
        observations = dataset["observation"]
        self.policy.eval()
        with torch.no_grad():
            pred = self.policy(observations)
        gt = dataset["action"]
        print("pred and gt", torch.cat([pred, gt], dim=-1), "error", torch.abs(pred - gt))
        abs_error = torch.abs(pred - gt)
        return torch.max(abs_error), torch.mean(abs_error)
        
    def parse_demo(self, demo_paths: List[str], save_image=False):
        transform = T.Compose([T.CenterCrop(224), T.ToTensor()])
        robot_model = toco.models.RobotModelPinocchio(
            "/home/yunfei/projects/fairo/polymetis/polymetis/data/franka_panda/panda_arm.urdf", "panda_link8"
        )
        all_obs, all_actions = [], []
        if save_image:
            import os
            import matplotlib.pyplot as plt
            if os.path.exists("tmp"):
                shutil.rmtree("tmp")
            os.makedirs("tmp")
            count = 0
        for file_name in demo_paths:
            print(f"Parsing {file_name}")
            with open(file_name, "rb") as f:
                while True:
                    try:
                        data = pickle.load(f)
                    except EOFError:
                        break
                    image = transform(Image.fromarray(data["image"].astype(np.uint8)))
                    image = image.reshape((-1, 3, 224, 224)).to(self.device)
                    if save_image:
                        plt.imsave("tmp/img%d.png" % count, image[0].permute(1, 2, 0).cpu().numpy())
                        count += 1
                    with torch.no_grad():
                        image_embedding = self.encode_fn(image * 255.0).squeeze(dim=0)
                    robot_state = data["robot_state"]
                    joint_positions = torch.Tensor(robot_state.joint_positions).to(self.device)
                    eef_pos, eef_quat = robot_model.forward_kinematics(joint_positions)
                    gripper_width = torch.Tensor([data["gripper_state"].width]).to(self.device)
                    desired_eef_pos = torch.Tensor(data["desired_eef_pos"]).to(self.device)
                    desired_eef_quat = torch.Tensor(data["desired_eef_quat"]).to(self.device)
                    desired_gripper = data.get("desired_gripper")
                    propriocep = torch.concat([joint_positions, eef_pos, gripper_width])
                    observation = torch.concat([image_embedding, propriocep])
                    if desired_gripper is None:
                        action = torch.concat([desired_eef_pos - eef_pos, torch.zeros(1, device=self.device)])
                    elif desired_gripper == "open":
                        action = torch.concat([desired_eef_pos - eef_pos, torch.ones(1, device=self.device)])
                    elif desired_gripper == "close":
                        action = torch.concat([desired_eef_pos - eef_pos, -torch.ones(1, device=self.device)])
                    action[:3] /= self.eef_scale
                    all_obs.append(observation)
                    all_actions.append(action)
            # for i in range(len(all_obs) - 1):
            #     # print(torch.norm(all_obs[i][:2048] - all_obs[-1][:2048]))
            #     print(torch.sum(all_obs[i][:2048] * all_obs[-1][:2048]) / (torch.norm(all_obs[i][:2048]) * torch.norm(all_obs[-1][:2048])))
        dataset = dict(
            observation=torch.stack(all_obs).float(),
            action=torch.stack(all_actions).float()
        )
        L = len(dataset["observation"])
        print(f"Parsing complete. Total number of samples: {L}")
        return dataset
