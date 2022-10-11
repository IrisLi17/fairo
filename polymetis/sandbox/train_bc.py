import argparse
from bc.bc_network import FCNetwork, DiscreteNetwork
from bc.trainer import BehaviorCloning
from r3m import load_r3m
import os
import torch


def main(args):
    encode_fn = load_r3m("resnet50")
    # control_net = FCNetwork(obs_dim=2048 + 11, act_dim=4, hidden_sizes=(256, 256))
    control_net = DiscreteNetwork(obs_dim=2048 + 11, act_dim=(3, 3, 3, 3), hidden_sizes=(256, 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encode_fn.to(device)
    control_net.to(device)
    encode_fn.eval()
    trainer = BehaviorCloning(control_net, encode_fn, device, lr=3e-4)
    expert_demos = []
    for i in range(5):
        expert_demos.append(os.path.join("/home/yunfei/Documents/demos", f"demo_pick{i+1}.pkl"))
    if not args.eval:
        trainer.train(expert_demos, num_epochs=500, batch_size=32)
    else:
        save_obj = torch.load(args.model_path, map_location=device)
        control_net.load_state_dict(save_obj["model"])
        max_error, mean_error = trainer.evaluate(expert_demos[1: 2])
        print("max error", max_error, "mean error", mean_error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
