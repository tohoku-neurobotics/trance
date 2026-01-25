#!/usr/bin/env bash
set -e

source ~/anaconda3/etc/profile.d/conda.sh
# If you’re currently in a conda env, deactivate it (optional)
conda deactivate || true
cd ~/ws_g1_slam
source devel/setup.bash
roslaunch livox_ros_driver2 msg_MID360.launch


