#!/usr/bin/env python3
import socket
import select
from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import String,  Int8, Int16, Empty, Float64
from sensor_msgs.msg import NavSatFix
from sailbot_msgs.msg import Wind
import grpc
from concurrent import futures
import json
import math
import time
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

import telemetry_messages.python.boat_state_pb2 as boat_state_pb2
import telemetry_messages.python.boat_state_pb2_grpc as boat_state_pb2_rpc
import telemetry_messages.python.control_pb2 as control_pb2
import telemetry_messages.python.control_pb2_grpc as control_pb2_grpc
import telemetry_messages.python.node_restart_pb2 as node_restart_pb2
import telemetry_messages.python.node_restart_pb2_grpc as node_restart_pb2_grpc

def make_json_string(json_msg):
    json_str = json.dumps(json_msg)
    message = String()
    message.data = json_str
    return message

class NetworkComms(Node):

    current_boat_state = boat_state_pb2.BoatState()

    def __init__(self):
        super().__init__('control_system')

        self.pwm_control_publisher_ = self.create_publisher(String, 'pwm_control', 10)
        self.trim_tab_control_publisher_ = self.create_publisher(Int8, 'tt_control', 10)
        self.trim_tab_angle_publisher_ = self.create_publisher(Int16, 'tt_angle', 10)

        self.rot_subscription = self.create_subscription(
            Float64,
            'airmar_data/rate_of_turn',
            self.rate_of_turn_callback,
            10)
        
        self.navsat_subscription = self.create_subscription(
            NavSatFix,
            'airmar_data/lat_long',
            self.lat_long_callback,
            10)
        
        self.track_degrees_true_subscription = self.create_subscription(
            Float64,
            'airmar_data/track_degrees_true',
            self.track_degrees_true_callback,
            10)
        
        self.track_degrees_magnetic_subscription = self.create_subscription(
            Float64,
            'airmar_data/track_degrees_magnetic',
            self.track_degrees_magnetic_callback,
            10)
        
        self.speed_knots_subscription = self.create_subscription(
            Float64,
            'airmar_data/speed_knots',
            self.speed_knots_callback,
            10)
        
        self.speed_knots_subscription = self.create_subscription(
            Float64,
            'airmar_data/speed_kmh',
            self.speed_kmh_callback,
            10)
        
        self.heading_subscription = self.create_subscription(
            Float64,
            'airmar_data/heading',
            self.heading_callback,
            10)
        
        self.true_wind_subscription = self.create_subscription(
            Wind,
            'airmar_data/true_wind',
            self.true_wind_callback,
            10)
        
        self.apparent_wind_subscription = self.create_subscription(
            Wind,
            'airmar_data/apparent_wind',
            self.apparent_wind_callback,
            10)
        
        self.roll_subscription = self.create_subscription(
            Float64,
            'airmar_data/roll',
            self.roll_callback,
            10)
        
        self.pitch_subscription = self.create_subscription(
            Float64,
            'airmar_data/pitch',
            self.pitch_callback,
            10)

        #initial dummy values, for testing
        self.current_boat_state.latitude = 5
        self.current_boat_state.longitude = 4
        self.current_boat_state.current_heading = 12
        self.current_boat_state.track_degrees_true = 0
        self.current_boat_state.track_degrees_magnetic = 0
        self.current_boat_state.speed_knots = 12
        self.current_boat_state.speed_kmh = 12
        self.current_boat_state.rate_of_turn = 2
        self.current_boat_state.true_wind.speed = 12
        self.current_boat_state.true_wind.direction = 5
        self.current_boat_state.apparent_wind.speed = 4
        self.current_boat_state.apparent_wind.direction = 6
        self.current_boat_state.pitch = 3
        self.current_boat_state.roll = 2
        self.node_indices = {}
        node_names = ["airmar_reader", "battery_monitor", "control_system", "debug_interface", "network_comms", "pwm_controller", "serial_rc_receiver", "trim_tab_comms"]
        i=0
        for name in node_names:
            node_info = boat_state_pb2.NodeInfo()
            node_info.name = name
            node_info.status = boat_state_pb2.NodeStatus.NODE_STATUS_WARN
            node_info.info = ""
            self.current_boat_state.node_states.append(node_info)
            self.node_indices[name]=i
            i+=1
        self.current_boat_state.current_autonomous_mode = boat_state_pb2.AutonomousMode.AUTONOMOUS_MODE_NONE
        a = boat_state_pb2.Point()
        a.latitude = 5.1
        a.longitude = 4.1
        b=boat_state_pb2.Point()
        b.latitude = 5.2
        b.longitude = 4.1
        self.current_boat_state.current_path.points.append(a)
        self.current_boat_state.current_path.points.append(b)
        c = boat_state_pb2.Point()
        c.latitude = 4.9
        c.longitude = 3.9
        d=boat_state_pb2.Point()
        d.latitude = 4.8
        d.longitude = 3.9
        e=boat_state_pb2.Point()
        e.latitude = 4.7
        e.longitude = 3.8
        self.current_boat_state.previous_positions.points.append(c)
        self.current_boat_state.previous_positions.points.append(d)
        self.current_boat_state.previous_positions.points.append(e)
        
        self.last_pwm_heartbeat = -1
        self.last_ctrl_heartbeat = -1
        self.last_tt_heartbeat = -1
        self.create_grpc_server()

        # # Create the I2C bus
        # self.i2c = busio.I2C(board.SCL, board.SDA)

        # # Create the ADC object using the I2C bus
        # self.ads = ADS.ADS1015(self.i2c)

        # # Create single-ended input on channel 0
        # self.chan = AnalogIn(self.ads, ADS.P0)

        self.pwm_heartbeat_subscription = self.create_subscription(
            Empty,
            'heartbeat/pwm_controler',
            self.pwm_controller_heartbeat,
            1)
        
        self.control_system_subscription = self.create_subscription(
            Empty,
            'heartbeat/control_system',
            self.control_system_heartbeat,
            1)
        
        self.control_system_subscription = self.create_subscription(
            Empty,
            'heartbeat/trim_tab_comms',
            self.trim_tab_comms_heartbeat,
            1)
        
        

    def rate_of_turn_callback(self, msg: Float64):
        self.current_boat_state.rate_of_turn = msg.data

    def lat_long_callback(self, msg: NavSatFix):
        self.current_boat_state.latitude = msg.latitude
        self.current_boat_state.longitude = msg.longitude

    def track_degrees_true_callback(self, msg: Float64):
        self.current_boat_state.track_degrees_true = msg.data

    def track_degrees_magnetic_callback(self, msg: Float64):
        self.current_boat_state.track_degrees_magnetic = msg.data

    def speed_knots_callback(self, msg: Float64):
        self.current_boat_state.speed_knots = msg.data

    def speed_kmh_callback(self, msg: Float64):
        self.current_boat_state.speed_kmh = msg.data
    
    def heading_callback(self, msg: Float64):
        self.current_boat_state.current_heading = msg.data

    def true_wind_callback(self, msg: Wind):
        self.current_boat_state.true_wind.speed = msg.speed.data
        self.current_boat_state.true_wind.direction = msg.direction.data

    def apparent_wind_callback(self, msg: Wind):
        self.current_boat_state.apparent_wind.speed = msg.speed.data
        self.current_boat_state.apparent_wind.direction = msg.direction.data

    def roll_callback(self, msg: Float64):
        self.current_boat_state.roll = msg.data
    
    def pitch_callback(self, msg: Float64):
        self.current_boat_state.pitch = msg.data

    #new server code
    def create_grpc_server(self): 
        self.get_logger().info("Creating gRPC server")
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        control_pb2_grpc.add_ExecuteRudderCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteTrimTabCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteBallastCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteAutonomousModeCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteSetPathCommandServiceServicer_to_server(self, self.grpc_server)
        boat_state_pb2_rpc.add_SendBoatStateServiceServicer_to_server(self, self.grpc_server)
        node_restart_pb2_grpc.add_RestartNodeServiceServicer_to_server(self, self.grpc_server)

        #connect_pb2_grpc.add_ConnectToBoatServiceServicer_to_server(self, self.grpc_server)
        self.grpc_server.add_insecure_port('[::]:50051')
        self.grpc_server.start()

    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteRudderCommand(self, command: control_pb2.RudderCommand, context):
        #center = 75 degrees, full right=40, full left = 113
        response = control_pb2.ControlResponse()
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_ERROR
        #rudder commands are radians, map to degrees
        degrees = int(command.rudder_control_value*(180/math.pi))
        degrees_scaled =  (((degrees - -90) * (113 - 40)) / (90 - -90)) + 40
        #self.get_logger().info("input angle: {}", degrees)
        rudder_json = {"channel": "8", "angle": degrees_scaled}
        #self.get_logger().info("Publishing rudder command: "+str(degrees))
        self.pwm_control_publisher_.publish(make_json_string(rudder_json))
        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteTrimTabCommand(self, command: control_pb2.TrimTabCommand, context):
        response = control_pb2.ControlResponse()
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_ERROR
        state_msg = Int8()
        state_msg.data = 5
        angle_msg = Int16()
        angle_msg.data = int((command.trimtab_control_value+math.pi/2)*180/math.pi)
        self.trim_tab_control_publisher_.publish(state_msg)
        self.trim_tab_angle_publisher_.publish(angle_msg)
        self.get_logger().info("Publishing trimtab command")
        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteBallastCommand(self, command: control_pb2.BallastCommand, context):
        response = control_pb2.ControlResponse()
        #self.get_logger().info("Publishing ballast command")
        value = 80 + ((110 - 80) / (1.0 - -1.0)) * (command.ballast_control_value - -1.0)
        ballast_json = {"channel": "12", "angle": value}
        self.pwm_control_publisher_.publish(make_json_string(ballast_json))
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_ERROR
        #self.get_logger.info("ADC read: {:>5}\t{:>5.3f}".format(self.chan.value, self.chan.voltage))
        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteAutonomousModeCommand(self, command: control_pb2.AutonomousModeCommand, context):
        response = control_pb2.ControlResponse()
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_ERROR
        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteSetPathCommand(self, command: control_pb2.SetPathCommand, context):
        response = control_pb2.ControlResponse()
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_ERROR
        self.current_boat_state.current_path.ClearField("points")# = command.new_path
        self.current_boat_state.current_path.points.extend(command.new_path.points)
        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def SendBoatState(self, command: boat_state_pb2.BoatStateRequest(), context):
        return self.current_boat_state
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def RestartNode(self, command: node_restart_pb2.RestartNodeRequest(), context):
        response = node_restart_pb2.RestartNodeResponse()
        response.success = False
        return response
    
    def pwm_controller_heartbeat(self, message):
        self.get_logger().info("Got pwm heartbeat")
        self.last_pwm_heartbeat = time.time()

    def control_system_heartbeat(self, message):
        self.get_logger().info("Got control heartbeat")
        self.last_ctrl_heartbeat = time.time()

    def trim_tab_comms_heartbeat(self, message):
        #self.get_logger().info("Got trimtab heartbeat")
        self.last_tt_heartbeat = time.time()
    
    def update_node_status_timer_callback(self):
        current_time = time.time()
        if(current_time-self.last_pwm_heartbeat>1):
            self.current_boat_state.node_states[self.node_indices["pwm_controller"]].node_status = boat_state_pb2.NodeStatus.NODE_STATUS_ERROR
        else:
            self.current_boat_state.node_states[self.node_indices["pwm_controller"]].node_status = boat_state_pb2.NodeStatus.NODE_STATUS_OK
        
        if(current_time-self.last_ctrl_heartbeat>1):
            self.current_boat_state.node_states[self.node_indices["control_system"]].node_status = boat_state_pb2.NodeStatus.NODE_STATUS_ERROR
        else:
            self.current_boat_state.node_states[self.node_indices["control_system"]].node_status = boat_state_pb2.NodeStatus.NODE_STATUS_OK


def main(args=None):
    rclpy.init(args=args)
    network_comms = NetworkComms()
    rclpy.spin(network_comms)
    rclpy.shutdown()

if __name__ == "__main__":
    main()