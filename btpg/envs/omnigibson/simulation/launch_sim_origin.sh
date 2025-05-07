#!/bin/bash

current_folder=$(cd "$(dirname "$0")";pwd)

isaacsim_path=$current_folder/../../../../simulators/omnigibson/IsaacSim4.2

echo ISAACSIM_PATH: $isaacsim_path
cd $isaacsim_path

./isaac-sim.sh --/isaac/startup/ros_bridge_extension= --/rtx/ecoMode/enabled=True \
