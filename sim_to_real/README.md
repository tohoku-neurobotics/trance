# Environment
- ubuntu 20.04, ros-noetic
- Pytorch
- Cuda version 12.8, driver-570
- Numpy 1.24.
- GPU, NVIDIA GTX 1650ti

# install unitree sdk2 for python
refer to the official [repo](https://github.com/unitreerobotics/unitree_sdk2_python)

# install zed camera sdk
- `sudo apt install zstd`
- download [sdk](https://www.stereolabs.com/en-jp/developers/release/5.0), version 5.0 for ubuntu 20.04
- `sudo chmod 777 your_sdk.run`
- `./your_sdk.run`
- `python -m pip install cython opencv-python pyopengl`
- `pip install requests`
- `cd /usr/local/zed`
- `python get_python_api.py`

# pytorch install
`pip3 install torch torchvision`, cuda 12.8 version

# for livox lidar slam
- Install LLVM C++ Standard library by following command, `sudo apt install libc++-dev libc++abi-dev`
- Install Eigen by following command, `sudo apt-get install libeigen3-dev`
- Install [Livox-SDK](https://github.com/Livox-SDK/Livox-SDK2)
- `cd ~/ws_g1_slam/src/FAST_LIO_LOCALIZATION_HUMANOID/livox_ros_driver2`
- `source /opt/ros/noetic/setup.sh`, `./build.sh ROS1`

# run each module in each terminal
- `./bash_1.sh`, roscore
- `./bash_2.sh`, locomotion
- `./bash_3.sh`, lidar scan and imu
- `./bash_4.sh`, lidar slam
- `./bash_5.sh`, camera detection
- `./bash_6.sh`, lidar map reconstruction and latent features
- `./bash_7.sh`, navigation


