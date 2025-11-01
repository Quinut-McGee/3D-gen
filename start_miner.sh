#!/bin/bash
cd /home/kobe/404-gen/v1/3D-gen
exec /home/kobe/miniconda3/envs/three-gen-mining/bin/python neurons/miner/competitive_miner.py "$@"
