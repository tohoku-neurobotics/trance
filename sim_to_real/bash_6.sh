#!/usr/bin/env bash
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch
python node_third_navigation_env.py

