#!/bin/bash
echo "Sourcing environment"
source /opt/ros/humble/setup.bash
source /home/sailbot/ros2_ws/install/local_setup.bash
source /home/sailbot/sailbot24-25/sailbot_ws/install/local_setup.bash
echo "Launching server"
python3 /opt/sailbot/startup_server.py
