#!/bin/bash
echo "Sourcing environment"
source /opt/ros/humble/setup.bash
source /usr/bin/python3
source /home/sailbot/ros2_ws/install/local_setup.bash
source /home/sailbot/sailbot24-25/sailbot_ws/install/local_setup.bash
echo "Launching server"
modprobe can
modprobe can_raw
modprobe mttcan
sudo ip link set can0 up type can bitrate 1000000
python3 /home/sailbot/sailbot24-25/startup/startup_server.py #/opt/sailbot/startup_server.py
