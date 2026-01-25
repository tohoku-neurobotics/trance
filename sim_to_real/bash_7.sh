#!/usr/bin/env bash
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch


python go2_navigation.py