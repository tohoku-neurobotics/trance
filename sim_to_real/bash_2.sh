#!/usr/bin/env bash
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch
python node_first_locomotion_env.py

