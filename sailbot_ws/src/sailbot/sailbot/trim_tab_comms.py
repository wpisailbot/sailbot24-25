#!/usr/bin/env python3
import struct
import asyncio
import os

import rclpy
from typing import Optional
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.timer import Timer
from rclpy.subscription import Subscription
from enum import Enum
from time import time as get_time

from std_msgs.msg import Int8, Int16, Float32, Empty
from sailbot_msgs.msg import Wind, AutonomousMode

import trim_tab_messages.python.messages_pb2 as message_pb2

import serial
import json
import time

serial_port = '/dev/ttyTHS0'
baud_rate = 115200 

# Local variables
state = message_pb2.TRIM_STATE.TRIM_STATE_MIN_LIFT
angle = 0
wind_dir = 0.0
battery_level = 100


class TrimTabComms(LifecycleNode):
    last_websocket = None
    last_winds = []
    autonomous_mode = 0

    def __init__(self):
        super(TrimTabComms, self).__init__('trim_tab_comms')

        self.tt_telemetry_publisher: Optional[Publisher]
        self.tt_battery_publisher: Optional[Publisher]
        self.tt_control_subscriber: Optional[Subscription]
        self.tt_angle_subscriber: Optional[Subscription]
        self.rudder_angle_subscriber: Optional[Subscription]
        self.ballast_pos_publisher: Optional[Publisher]

        self.autonomous_mode_subscriber: Optional[Subscription]


        self.timer_pub: Optional[Publisher]

        self.heartbeat_timer: Optional[Timer]
        self.timer: Optional[Timer]

    #lifecycle node callbacks
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("In configure")
        self.tt_telemetry_publisher = self.create_lifecycle_publisher(Float32, 'tt_telemetry', 10)
        self.ballast_pos_publisher = self.create_lifecycle_publisher(Int16, 'current_ballast_position', 10)

        self.tt_battery_publisher = self.create_lifecycle_publisher(Int8, 'tt_battery', 10)  # Battery level
        self.tt_control_subscriber = self.create_subscription(Int8, 'tt_control', self.tt_state_callback, 10)  # Trim tab state
        self.tt_angle_subscriber = self.create_subscription(Int16, 'tt_angle', self.tt_angle_callback, 10)

        self.rudder_angle_subscriber = self.create_subscription(Int16, 'rudder_angle', self.rudder_angle_callback, 10)
        self.ballast_pwm_subscriber = self.create_subscription(Int16, 'ballast_pwm', self.ballast_pwm_callback, 10)

        self.apparent_wind_publisher = self.create_subscription(Wind, 'airmar_data/apparent_wind', self.apparent_wind_callback, 10)

        self.autonomous_mode_subscriber = self.create_subscription(AutonomousMode, 'autonomous_mode', self.autonomous_mode_callback, 10)

        self.timer_pub = self.create_lifecycle_publisher(Empty, '/heartbeat/trim_tab_comms', 1)
        
        self.ballast_timer = self.create_timer(0.01, self.ballast_timer_callback)

        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=0.05)
        except Exception as e:
            self.get_logger().info(str(e))
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating...")
        # Start publishers or timers
        return super().on_activate(state)


    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating...")
        return super().on_deactivate(state)
        

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up...")
        # Destroy subscribers, publishers, and timers
        self.destroy_lifecycle_publisher(self.tt_telemetry_publisher)
        self.destroy_lifecycle_publisher(self.tt_battery_publisher)
        self.destroy_lifecycle_publisher(self.ballast_pos_publisher)
        self.destroy_lifecycle_publisher(self.timer_pub)
        self.destroy_subscription(self.tt_control_subscriber)
        self.destroy_subscription(self.tt_angle_subscriber)
        self.destroy_timer(self.heartbeat_timer)
        self.destroy_timer(self.ballast_timer)

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down...")
        # Perform final cleanup if necessary
        return TransitionCallbackReturn.SUCCESS
    
    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Error caught!")
        return super().on_error(state)
    
    #end callbacks
    def autonomous_mode_callback(self, msg: AutonomousMode):
        self.get_logger().info(f"Got autonomous mode: {msg.mode}")
        self.autonomous_mode = msg.mode

    def apparent_wind_callback(self, msg: Wind):
        direction = msg.direction
        self.find_trim_tab_state(direction)

    def update_winds(self, relative_wind):
        # Check we have new wind
        if len(self.last_winds) != 0 and relative_wind == self.last_winds[len(self.last_winds) - 1]:
            return
            # First add wind to running list
        self.last_winds.append(float(relative_wind))
        if len(self.last_winds) > 10:
            self.last_winds.pop(0)
        # Now find best trim tab state
        smooth_angle = self.median(self.last_winds)
        return smooth_angle

    def find_trim_tab_state(self, relative_wind):  # five states of trim
        smooth_angle = self.update_winds(relative_wind)
        
        # Check autonomous mode TODO: This is a coupling that shouldn't be necessary. 
        # Can be fixed by separating nodes and using lifecycle state transitions, or by finishing behavior tree
        autonomous_modes = AutonomousMode()
        if (self.autonomous_mode != autonomous_modes.AUTONOMOUS_MODE_FULL or self.autonomous_modes != autonomous_modes.AUTONOMOUS_MODE_TRIMTAB):
            return

        msg = None
        if 45.0 <= smooth_angle < 135:
            # Max lift port
            msg = {
                state: "max_lift_port"
            }
        elif 135 <= smooth_angle < 180:
            # Max drag port
            msg = {
                state: "max_drag_port"
            }
        elif 180 <= smooth_angle < 225:
            # Max drag starboard
            msg = {
                state: "max_drag_starboard"
            }
        elif 225 <= smooth_angle < 315:
            # Max lift starboard
            msg = {
                state: "max_lift_starboard"
            }
        else:
            # In irons, min lift
            msg = {
                state: "min_lift"
            }
        message_string = json.dumps(msg)+'\n'
        self.ser.write(message_string.encode())

    def tt_state_callback(self, msg: Int8):
        protomsg = message_pb2.ControlMessage()
        protomsg.control_type = message_pb2.CONTROL_MESSAGE_CONTROL_TYPE.CONTROL_MESSAGE_CONTROL_TYPE_STATE
        protomsg.state = msg.data
        serialized_message = protomsg.SerializeToString()
        if(self.last_websocket is not None):
            self.schedule_async_function(self.last_websocket.send(serialized_message))

    def tt_angle_callback(self, msg: Int16):
        self.get_logger().info("Sending trimtab angle")
        angle = msg.data
        this_time = get_time()
        message = {
            "state": "manual",
            "angle": angle,
            "timestamp": this_time
        }
        message_string = json.dumps(message)+'\n'
        self.ser.write(message_string.encode())

    def rudder_angle_callback(self, msg: Int16):
        self.get_logger().info(f"Got rudder position: {msg.data}")
        degrees = msg.data+13 #Servo degree offset

        message = {
            "rudder_angle": degrees
        }
        message_string = json.dumps(message)+'\n'
        self.ser.write(message_string.encode())

    def ballast_pwm_callback(self, msg: Int16):
        #self.get_logger().info("Got ballast position")
        pwm = msg.data
        message = {
            "ballast_pwm": pwm
        }
        message_string = json.dumps(message)+'\n'
        self.ser.write(message_string.encode())

    def ballast_timer_callback(self):
        message = {
            "get_ballast_pos": True
        }
        message_string = json.dumps(message)+'\n'
        self.ser.write(message_string.encode())
        line = None
        try:
            line = self.ser.readline().decode('utf-8').rstrip()
        except:
            #serial corruption
            pass
        
        if line:
            try:
                message = json.loads(line)

                #self.get_logger().info("Received position:", message["ballast_pos"])
                pos = Int16()
                pos.data = message["ballast_pos"]
                self.ballast_pos_publisher.publish(pos)
            except json.JSONDecodeError:
                self.get_logger().info("Error decoding JSON")
        else:
            self.get_logger().info("No data received within the timeout period.")

    async def echo(self, websocket, path):
        self.last_websocket = websocket
        async for message in websocket:
            try:
                # Deserialize the protobuf message
                protobuf_message = message_pb2.DataMessage()
                protobuf_message.ParseFromString(message)

                self.get_logger().info("Received message:"+ str(protobuf_message.windAngle)+": "+str(protobuf_message.batteryLevel))

                # Optionally, send a response back (as a string or protobuf)
                await websocket.send("Message received")
                
            except Exception as e:
                self.get_logger().info("Error processing message:", e)
                raise(e)

def main(args=None):
    rclpy.init(args=args)

    loop = asyncio.get_event_loop()
    tt_comms = TrimTabComms()
    #for debugging purposes
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        ser.close()
    except Exception as e:
        tt_comms.get_logger().info(str(e))
    # Use the SingleThreadedExecutor to spin the node.
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(tt_comms)

    executor.spin()
    tt_comms.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
