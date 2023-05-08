![Polymetis: a real-time PyTorch controller manager](./docs/source/img/polymetis-logo.svg)

## A real-time PyTorch controller manager

[![CircleCI](https://circleci.com/gh/facebookresearch/fairo/tree/main.svg?style=svg&circle-token=7fadbd3989ab8e76003fd5193ad62e26686bc4a6)](https://circleci.com/gh/facebookresearch/fairo/tree/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Polymetis**: adj., Ancient Greek. _Clever in many ways._ Applied to Odysseus by Homer in the _Odyssey_.

**Write [PyTorch](http://pytorch.org/) controllers for robots, test them in simulation, and seamlessly transfer to real-time hardware.**

**Polymetis** powers robotics research at [Facebook AI Research](https://ai.facebook.com/). If you want to write your robot policies in PyTorch for simulation and immediately transfer them to high-frequency (1kHz) policies on real-time hardware (e.g. Franka Panda), read on!

## Features

- **Unified simulation & hardware interface**: Write all your robot controllers just once -- immediately transfer them to real-time hardware. You can even train neural network policies using reinforcement learning in simulation and transfer them to hardware, with just a single configuration toggle.
- **Write your own robot controllers:** Use the building blocks in our [TorchControl](https://facebookresearch.github.io/fairo/polymetis/torchcontrol-doc.html) library to write complex robot controllers, including operational space control. Take advantage of our wrapping of the [Pinocchio](https://github.com/stack-of-tasks/pinocchio) dynamics library for your robot dynamics.
- **Drop-in replacement for [PyRobot](https://pyrobot.org/)**: If you're already using PyRobot, you can use the exact same interface, but immediately gain access to arbitrary, custom high-frequency robot controllers.

## Get started

To install from source, you need to run the following lines on NUC (for control) and a GPU machine (for training/inference)
```
git clone git@git.tsinghua.edu.cn:liyf20/fairo.git
cd fairo/polymetis
# Create environment from appropriate files
# Only on NUC
conda env create -f ./polymetis/environment-cpu.yml
# Only on GPU
conda env create -f ./polymetis/environment.yml # Pytorch version is 1.10.0 with cudatoolkit 11.3, you can modify these versions to your needs, e.g., CLIPort requires pytorch 1.7 + cuda 11.0

conda activate polymetis-local
pip install -e ./polymetis

# Only required on NUC
./scripts/build_libfranka.sh 0.9.0

# Build from source
mkdir -p ./polymetis/build
cd ./polymetis/build
# Set -DBUILD_FRANKA=ON on NUC; set -DBUILD_FRANKA=OFF on GPU machine
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=[OFF/ON] -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF
make -j
```
(Disclaimer: Dependencies related to the camera and gamepad, such as `pyrealsense2`, `opencv-python`, `pygame` have not been updated in `*.yml`. You can install them via `pip`.)

After turning on the robot and activating FCI, you can start services on NUC with the following lines. Each line runs in a separate tab/terminal.
```
# Robot service requires sudo. The readonly=true can be used in applications like calibration where we want to make sure no control command is sent to the robot.
python polymetis/python/scripts/launch_robot.py robot_client=franka_hardware [robot_client.executable_cfg.readonly=true]
# Camera service.
python polymetis/python/scripts/launch_camera.py width=848 height=480 framerate=15 downsample=1 use_depth=[true/false] # Valid combination of resolution and framerate can be looked up from RealSense datasheets. 
# Gripper service.
python polymetis/python/scripts/launch_gripper.py gripper=franka_hand
``` 

Now you can move to the GPU machine and try talking with these services. For example, you can open up a python shell (when `polymetis-local` environment is activated) and try:
```
from polymetis import RobotInterface
robot_interface = RobotInterface(ip_address="10.100.7.16") # This was the IP of the NUC, please double check.
robot_interface.go_home()
```
Some example code snippets are in `examples`. You can read camera images with `python examples/camera_monitor.py <ip>`, play with grippers with `python examples/gripper_test.py [ip] [open/close]`.

## Camera calibration

Eye-on-base calibration: mount the calibration plate to the robot, make sure the camera is rigidly attached to the table/ground. Go to the robot webpage and set the end effector to `None`.
Start services in NUC as follows
```
# In Tab 1
python polymetis/python/scripts/launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.readonly=true
# In Tab 2
python polymetis/python/scripts/launch_camera.py width=1280 height=720 framerate=15 downsample=1 use_depth=false
```
Switch the robot to white.
Run calibration:
```
cd fairo/polymetis/examples
python calibration/calibration_service.py [--ip="xxx"] # set ip if this line is run from GPU machine
```
Change the robot pose, and press `c` to capture one sample. After collecting 4 samples, press `r` to compute the transformation. If the matrix looks good, press `s` to save the result to `calib_handeye.pkl`. Press `ctrl-c` to stop calibration. Rename the file before running another calibration. The recommended way to verify the calibration is repeating the process for 3 times then check the variance of `debug_info` stored in the `pkl` file. The variance should be ~0.001.

After calibration, replace the calibration plate with panda hand and remember to change the end effector setting in the webpage. 

## CLIPort demo collection
A pick-and-place demo collector is in `examples/cliport_demo.py`. You can try with `python examples/cliport_demo.py --ip=[ip] --folder_name=[folder_to_store_demonstrations] --calibration_file=[path_to_easyhandeye_calibration_yml]`. Hand-eye calibration results from the yml generated from easy_handeye (which by default locates under the folder `~/.ros/easy_hand_eye`) or the pkl generated from our calibration service can be applied.  

[The following part is from the official repo, which does not include cameras] To get started, you only need one line:

```
conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
```

You can immediately start running the [example scripts](https://github.com/facebookresearch/fairo/tree/main/polymetis/examples) in both simulation and hardware. See [installation](https://facebookresearch.github.io/fairo/polymetis/installation.html) and [usage](https://facebookresearch.github.io/fairo/polymetis/usage.html) documentation for details.

## Documentation

All documentation on the [website](https://facebookresearch.github.io/fairo/polymetis/). Includes:

- Guides on setting up your [Franka Panda](https://frankaemika.github.io/docs/libfranka.html) hardware for real-time control
- How to quickly get started in [PyBullet](https://github.com/bulletphysics/bullet30) simulation
- Writing developing your own custom controllers in PyTorch
- Full [autogenerated documentation](https://facebookresearch.github.io/fairo/polymetis/modules.html)

## Benchmarking

To run benchmarking, first configure the [script](polymetis/tests/python/polymetis/benchmarks/benchmark_robustness.py) to point to your hardware instance, then run

```bash
asv run --python=python --set-commit-hash $(git rev-parse HEAD)
```

To update the dashboard, run:

```bash
asv publish
```

Commit the result under `.asv/results` and `docs/`; it will show up under the benchmarking page in the documentation.

## Citing
If you use Polymetis in your research, please use the following BibTeX entry.
```
@misc{Polymetis2021,
  author =       {Lin, Yixin and Wang, Austin S. and Sutanto, Giovanni and Rai, Akshara and Meier, Franziska},
  title =        {Polymetis},
  howpublished = {\url{https://facebookresearch.github.io/fairo/polymetis/}},
  year =         {2021}
}
```

Note: Giovanni Sutanto contributed to the repository during his research internship at Facebook Artificial Intelligence Research (FAIR) in Fall 2019.

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out. [Make an issue](https://github.com/facebookresearch/fairo/issues/new/choose) for bugs and feature requests, or contribute a new robot controller by making a [pull request](https://github.com/facebookresearch/fairo/pulls)!

## License
Polymetis is MIT licensed, as found in the [LICENSE](LICENSE) file.
