from bc.bc_network import FCNetwork
from bc.trainer import BehaviorCloning
from r3m import load_r3m
import os
import torch


def main():
    encode_fn = load_r3m("resnet50")
    control_net = FCNetwork(obs_dim=2048 + 11, act_dim=4, hidden_sizes=(256, 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encode_fn.to(device)
    control_net.to(device)
    encode_fn.eval()
    trainer = BehaviorCloning(control_net, encode_fn, device, lr=1e-3)
    trainer.train([os.path.join(os.path.dirname(__file__), "..", "demo_first.pkl")], num_epochs=100, batch_size=32)

if __name__ == "__main__":
    main()
