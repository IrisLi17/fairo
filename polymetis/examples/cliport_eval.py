"""Ravens main training script."""

import os
import pickle
import json

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport.utils import utils
from polymetis import CameraInterface, RobotInterface, GripperInterface
import torchcontrol.transform.rotation as rotation
import torch
import time


# @hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):
    camera_interface = CameraInterface(ip_address=vcfg["ip"])
    robot_interface = RobotInterface(ip_address=vcfg["ip"])
    gripper_interface = GripperInterface(ip_address=vcfg["ip"])
    robot_interface.go_home()
    tcfg = vcfg["train_config"]


    # Choose eval mode and task.
    # mode = vcfg['mode']
    eval_task = vcfg['eval_task']
    # if mode not in {'train', 'val', 'test'}:
    #     raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    # dataset_type = vcfg['type']
    ds = dataset.RealDataset(vcfg['data_dir'], tcfg)
    # if 'multi' in dataset_type:
    #     ds = dataset.RavensMultiTaskDataset(vcfg['data_dir'],
    #                                         tcfg,
    #                                         group=eval_task,
    #                                         mode=mode,
    #                                         n_demos=vcfg['n_demos'],
    #                                         augment=False)
    # else:
    #     ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
    #                                tcfg,
    #                                n_demos=vcfg['n_demos'],
    #                                augment=False)

    all_results = {}
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])

    # Save path for results.
    # json_name = f"multi-results-{mode}.json" if 'multi' in vcfg['model_path'] else f"results-{mode}.json"
    # save_path = vcfg['save_path']
    # print(f"Save path for results: {save_path}")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # save_json = os.path.join(save_path, f'{name}-{json_name}')

    # Load existing results.
    # existing_results = {}
    # if os.path.exists(save_json):
    #     with open(save_json, 'r') as f:
    #         existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    # ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)
    ckpts_to_eval = [vcfg["checkpoint"]]

    # Evaluation loop
    print(f"Evaluating: {str(ckpts_to_eval)}")
    for ckpt in ckpts_to_eval:
        model_file = ckpt

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        # elif not vcfg['update_results'] and ckpt in existing_results:
        #     print(f"Skipping because of existing results for {model_file}.")
        #     continue

        results = []
        mean_reward = 0.0

        # Initialize agent.
        # utils.set_seed(train_run, torch=True)
        agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

        # Load checkpoint
        agent.load(model_file)
        print(f"Loaded: {model_file}")

        # record = vcfg['record']['save_video']
        n_demos = vcfg['n_demos']

        # Run testing and save total rewards with last transition info.
        for i in range(0, n_demos):
            print(f'Test: {i + 1}/{n_demos}')
            # episode, seed = ds.load(i)
            # goal = episode[-1]
            # total_reward = 0
            # np.random.seed(seed)

            # set task
            # if 'multi' in dataset_type:
            #     task_name = ds.get_curr_task()
            #     task = tasks.names[task_name]()
            #     print(f'Evaluating on {task_name}')
            # else:
            #     task_name = vcfg['eval_task']
            #     task = tasks.names[task_name]()

            # task.mode = mode
            # env.seed(seed)
            # env.set_task(task)
            # reward = 0

            quat = torch.Tensor([0.9211, -0.3892,  0.0022, -0.0074])
            quat = (rotation.from_quat(torch.Tensor([0., 0., np.sin(np.pi / 2 / 2), np.cos(np.pi / 2 / 2)])) * rotation.from_quat(quat)).as_quat()

            # Start recording video (NOTE: super slow)
            # if record:
            #     video_name = f'{task_name}-{i+1:06d}'
            #     if 'multi' in vcfg['model_task']:
            #         video_name = f"{vcfg['model_task']}-{video_name}"
            #     env.start_rec(video_name)

            for _ in range(vcfg['max_steps']):
                im_count = 0
                old_stamp = None
                depth_buffer = []
                while im_count < 15:
                    img, stamp = camera_interface.read_once()
                    if stamp != old_stamp:
                        im_count += 1
                        rgb = img[..., :3]
                        depth_buffer.append(img[..., -1])
                        old_stamp = stamp
                depth_img = np.median(np.stack(depth_buffer, axis=0), axis=0)
                obs = {"color": [rgb], "depth": [depth_img]}
                # obs = env.reset()
                # info = env.info
                lang_goal = input("Enter language goal:")
                info = {'lang_goal': lang_goal}
                print(f'Lang Goal: {lang_goal}')
                act = agent.act(obs, info, goal=None)
                p0_xyz, p0_xyzw = act["pose0"]
                p1_xyz, p1_xyzw = act["pose1"]
                p0 = torch.Tensor(p0_xyz)
                p1 = torch.Tensor(p1_xyz)
                quat0 = quat
                quat1 = quat
                # convert to link8 position
                p0[2] += 0.1034
                p1[2] += 0.1034
                # Since p0 and p1 are surface coordinates, convert to robot pose here
                p0[2] -= 0.01
                p1[2] += 0.05 - 0.01
                execute_pnp(robot_interface, gripper_interface, p0, quat0, p1, quat1)
                is_done = input("Shall we terminate? [y/other]")
                if is_done == "y":
                    break
                # obs, reward, done, info = env.step(act)
                # total_reward += reward
                # print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
                # if done:
                #     break

            # results.append((total_reward, info))
            # mean_reward = np.mean([r for r, i in results])
            # print(f'Mean: {mean_reward} | Task: {task_name} | Ckpt: {ckpt}')

            # End recording video
            # if record:
            #     env.end_rec()

        all_results[ckpt] = {
            'episodes': results,
            'mean_reward': mean_reward,
        }

        # Save results in a json file.
        # if vcfg['save_results']:

        #     # Load existing results
        #     if os.path.exists(save_json):
        #         with open(save_json, 'r') as f:
        #             existing_results = json.load(f)
        #         existing_results.update(all_results)
        #         all_results = existing_results

        #     with open(save_json, 'w') as f:
        #         json.dump(all_results, f, indent=4)


def execute_pnp(robot_interface, gripper_interface, p0, quat0, p1, quat1):
    approach_pos = p0 + torch.Tensor([0, 0, 0.05])
    robot_interface.move_to_ee_pose(approach_pos, quat0)
    pick_traj = robot_interface.move_to_ee_pose(p0, quat0)
    # pick
    gripper_interface.grasp(speed=0.1, force=5)
    time.sleep(1)
    while gripper_interface.get_state().is_moving:
        time.sleep(0.1)
    robot_interface.move_to_ee_pose(approach_pos, quat0)
    place_approach_pos = p1 + torch.Tensor([0, 0, 0.05])
    robot_interface.move_to_ee_pose(place_approach_pos, quat1, op_space_interp=True)
    placement_traj = robot_interface.move_to_ee_pose(p1, quat1)
    # release
    gripper_interface.goto(width=0.08, speed=0.1, force=1)
    time.sleep(1)
    while gripper_interface.get_state().is_moving:
        time.sleep(1)
    robot_interface.move_to_ee_pose(place_approach_pos, quat1)
    # reset
    robot_interface.go_home()

def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg['checkpoint_type'] == 'last':
        last_ckpt = 'last.ckpt'
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg['checkpoint_type'] == 'val_missing':
        checkpoints = sorted([c for c in os.listdir(vcfg['model_path']) if "steps=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg['checkpoint_type'] == 'test_best':
        result_jsons = [c for c in os.listdir(vcfg['results_path']) if "results-val" in c]
        if 'multi' in vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            best_checkpoint = 'last.ckpt'
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res['mean_reward'] > best_success:
                    best_checkpoint = ckpt
                    best_success = res['mean_reward']
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = 'last.ckpt'
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        checkpoints = [c for c in os.listdir(vcfg['model_path']) if vcfg['checkpoint_type'] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    return ckpts_to_eval


def test(vcfg):
    import matplotlib.pyplot as plt
    tcfg = vcfg['train_config']
    name = ""
    ds = dataset.RealDataset(vcfg['data_dir'], tcfg)
    agent = agents.names[vcfg['agent']](name, tcfg, None, ds)
    agent.load(vcfg["checkpoint"])
    folder_name = "/home/yunfei/Documents/cliport_demo/train"
    for fname in os.listdir(folder_name):
        full_name = os.path.join(folder_name, fname)
        with open(full_name, "rb") as f:
            data = pickle.load(f)
        new_lang = input(data["lang_goal"])
        obs = {"color": [data["obs"][0]["color"]], "depth": [data["obs"][0]["depth"]]}
        cmap = ds.get_image(obs)
        # import matplotlib.pyplot as plt
        # plt.imsave("tmp.png", cmap[..., :3].astype(np.uint8))
        info = {"lang_goal": new_lang}
        print("info", info)
        act = agent.act(obs, info, None)
        print("act", act)
        print("gt", data["action"])
        vis_img = np.concatenate([cmap[..., :3], 128 * np.ones((cmap.shape[0], cmap.shape[1], 1))], axis=-1).astype(np.uint8)
        vis_img[act["pick"][0] - 2: act["pick"][0] + 3, act["pick"][1] - 2: act["pick"][1] + 3] = np.array([255, 0, 255, 255], dtype=np.uint8)
        gt_p0 = utils.xyz_to_pix(data["action"][0]["p0"][0], ds.bounds, ds.pix_size)
        vis_img[gt_p0[0] - 2: gt_p0[0] + 3, gt_p0[1] - 2: gt_p0[1] + 3] = np.array([255, 255, 0, 255], dtype=np.uint8)
        plt.imsave("tmp.png", vis_img)

if __name__ == '__main__':
    tcfg = dict(
        dataset={'images': True, 'cache': True},
        train=dict(
            task="real",
            agent="two_stream_full_clip_lingunet_lat_transporter",
            n_rotations=36,
            batchnorm=False, # important: False because batch_size=1
            lr=1e-4,
            # script configs
            gpu=[0], # -1 for all
            log=False, # log metrics and stats to wandb
            attn_stream_fusion_type='add',
            trans_stream_fusion_type='conv',
            lang_fusion_type='mult',
            n_val=6,
            val_repeats=1,
            save_steps=[1000, 2000, 3000, 4000, 5000, 7000, 10000, 20000, 40000, 80000, 120000, 160000, 200000, 300000, 400000, 500000, 600000, 800000, 1000000, 1200000],
            load_from_last_ckpt=True,
        )
    )
  
    vcfg = dict(
        ip="101.6.103.171",
        data_dir="./",
        eval_task="real",
        agent="cliport",
        n_demos=1,
        max_steps=3,
        checkpoint="/home/yunfei/projects/revised_cliport/steps=02000-val_loss=0.00131132.ckpt",
        train_config=tcfg,
    )
    main(vcfg)
    # test(vcfg)