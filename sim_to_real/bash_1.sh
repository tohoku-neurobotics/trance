#!/usr/bin/env bash
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate || true
roscore


