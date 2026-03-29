#!/bin/bash
cd /home/bolongli/Sailbot/sailbot24-25
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source /home/bolongli/Sailbot/sailbot24-25/sailbot_ws/install/setup.bash
exec python3 /home/bolongli/Sailbot/sailbot24-25/sailbot_ws/src/sailbot/sailbot/buoy_detection_yolo.py "$@"