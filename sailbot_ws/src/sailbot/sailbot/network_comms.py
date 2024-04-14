#!/usr/bin/env python3
import sys
import rclpy
from typing import Optional
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.timer import Timer
from rclpy.subscription import Subscription
from std_msgs.msg import String,  Int8, Int16, Empty, Float64
from lifecycle_msgs.msg import TransitionEvent
from lifecycle_msgs.msg import State as StateMsg
from sensor_msgs.msg import NavSatFix, Image
from geographic_msgs.msg import GeoPoint
from nav_msgs.msg import OccupancyGrid
from ament_index_python.packages import get_package_share_directory
from sailbot_msgs.msg import Wind, Path, AutonomousMode, TrimState, Waypoint, WaypointPath
import grpc
from concurrent import futures
import json
import math
import time
import os
import numpy as np
import cv2
import re
import numpy as np
import traceback
import types
from typing import Callable, Any

from sailbot_msgs.srv import RestartNode

import telemetry_messages.python.boat_state_pb2 as boat_state_pb2
import telemetry_messages.python.boat_state_pb2_grpc as boat_state_pb2_rpc
import telemetry_messages.python.control_pb2 as control_pb2
import telemetry_messages.python.control_pb2_grpc as control_pb2_grpc
import telemetry_messages.python.node_restart_pb2 as node_restart_pb2
import telemetry_messages.python.node_restart_pb2_grpc as node_restart_pb2_grpc
import telemetry_messages.python.video_pb2 as video_pb2
import telemetry_messages.python.video_pb2_grpc as video_pb2_grpc

def find_and_load_image(directory, location):
    """
    Find an image by location and load it along with its bounding box coordinates.

    Parameters:
    - directory: The directory to search in.
    - location: The location to match.

    Returns:
    - A tuple containing the loaded image and a dictionary with the bounding box coordinates.
    """
    # Regular expression to match the filename format and capture coordinates
    pattern = re.compile(rf"{location}:(-?\d+\.?\d*):(-?\d+\.?\d*):(-?\d+\.?\d*):(-?\d+\.?\d*)\.png")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the bounding box coordinates
            south, west, north, east = map(float, match.groups())

            # Load the image
            image_path = os.path.join(directory, filename)
            image = 255-cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.flip(image, 0)
            img_rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
            alpha_channel = np.ones(image.shape, dtype=np.uint8) * 127
            alpha_channel[image == 255] = 0 
            img_rgba[:, :, 3] = alpha_channel
            if image is None:
                raise ValueError(f"Unable to load image at {image_path}")

            return img_rgba, {"south": south, "west": west, "north": north, "east": east}

    # Return None if no matching file is found
    return None, None

def get_resource_dir():
    package_path = get_package_share_directory('sailbot')
    resource_path = os.path.join(package_path, 'maps')
    return resource_path

def encode_frame(frame):
    # Convert the frame to JPEG
    result, encoded_image = cv2.imencode('.jpg', frame)
    if result:
        return encoded_image.tobytes()
    else:
        return None

def make_json_string(json_msg):
    json_str = json.dumps(json_msg)
    message = String()
    message.data = json_str
    return message

def get_state(state_id: int):
    if state_id == StateMsg.PRIMARY_STATE_ACTIVE:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_ACTIVE
    if state_id == StateMsg.PRIMARY_STATE_INACTIVE:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_INACTIVE
    if state_id == StateMsg.PRIMARY_STATE_FINALIZED:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_FINALIZED
    if state_id == StateMsg.PRIMARY_STATE_UNCONFIGURED:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_UNCONFIGURED
    if state_id == StateMsg.PRIMARY_STATE_UNKNOWN:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_UNKNOWN
    if state_id == StateMsg.TRANSITION_STATE_ACTIVATING:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_ACTIVATING
    if state_id == StateMsg.TRANSITION_STATE_CLEANINGUP:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_CLEANING_UP
    if state_id == StateMsg.TRANSITION_STATE_CONFIGURING:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_CONFIGURING
    if state_id == StateMsg.TRANSITION_STATE_DEACTIVATING:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_DEACTIVATING
    if state_id == StateMsg.TRANSITION_STATE_ERRORPROCESSING:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_ERROR_PROCESSING
    if state_id == StateMsg.TRANSITION_STATE_SHUTTINGDOWN:
        return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_SHUTTINGDOWN
    return boat_state_pb2.NodeLifecycleState.NODE_LIFECYCLE_STATE_UNKNOWN

class NetworkComms(LifecycleNode):

    current_map: OccupancyGrid = None
    current_boat_state = boat_state_pb2.BoatState()
    last_camera_frame = None
    last_camera_frame_shape = None
    last_camera_frame_time = time.time()
    do_video_encode = False
    current_video_source = 0

    def __init__(self):
        super().__init__('network_comms')
        self.rudder_control_publisher: Optional[Publisher]
        self.ballast_position_publisher: Optional[Publisher]
        self.trim_tab_control_publisher: Optional[Publisher]
        self.trim_tab_angle_publisher: Optional[Publisher]
        self.autonomous_mode_publisher: Optional[Publisher]
        self.waypoints_publisher: Optional[Publisher]
        self.single_waypoint_publisher: Optional[Publisher]

        self.rot_subscription: Optional[Subscription]
        self.navsat_subscription: Optional[Subscription]
        self.track_degrees_true_subscription: Optional[Subscription]
        self.track_degrees_magnetic_subscription: Optional[Subscription]
        self.speed_knots_subscription: Optional[Subscription]
        self.speed_kmh_subscription: Optional[Subscription]
        self.heading_subscription: Optional[Subscription]
        self.true_wind_subscription: Optional[Subscription]
        self.apparent_wind_subscription: Optional[Subscription]
        self.roll_subscription: Optional[Subscription]
        self.pitch_subscription: Optional[Subscription]
        self.pwm_heartbeat_subscription: Optional[Subscription]
        self.control_system_subscription: Optional[Subscription]
        self.current_path_subscription: Optional[Subscription]
        self.target_position_subscriber: Optional[Subscription]
        self.trim_state_subscriber: Optional[Subscription]
        self.camera_image_subscriber: Optional[Subscription]

        #receives state updates from other nodes
        self.airmar_reader_lifecycle_state_subscriber: Optional[Subscription]

        self.callback_group_state = MutuallyExclusiveCallbackGroup()

        self.declare_parameter('map_name', 'quinsigamond')
        self.map_name = self.get_parameter('map_name').get_parameter_value().string_value
        self.get_logger().info(f'Map name: {self.map_name}')
        self.get_logger().info("Getting map image")
        self.map_image, self.bbox = find_and_load_image(get_resource_dir(), self.map_name)
        
    #lifecycle node callbacks
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("In configure")
        self.pwm_control_publisher = self.create_lifecycle_publisher(String, 'pwm_control', 10)
        self.rudder_control_publisher = self.create_lifecycle_publisher(Int16, 'rudder_angle', 10)

        self.ballast_position_publisher = self.create_lifecycle_publisher(Float64, 'ballast_position', 10)
        self.trim_tab_control_publisher = self.create_lifecycle_publisher(Int8, 'tt_control', 10)
        self.trim_tab_angle_publisher = self.create_lifecycle_publisher(Int16, 'tt_angle', 10)

        self.waypoints_publisher = self.create_lifecycle_publisher(WaypointPath, 'waypoints', 10)
        self.single_waypoint_publisher = self.create_lifecycle_publisher(Waypoint, 'single_waypoint', 10)


        self.autonomous_mode_publisher = self.create_lifecycle_publisher(AutonomousMode, 'autonomous_mode', 10)

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
        
        self.speed_kmh_subscription = self.create_subscription(
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
            'true_wind_smoothed',
            self.true_wind_callback,
            10)
        
        self.apparent_wind_subscription = self.create_subscription(
            Wind,
            'apparent_wind_smoothed',
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

        self.current_path_subscription = self.create_subscription(
            Path,
            'current_path',
            self.current_path_callback,
            10)
        self.target_position_subscriber = self.create_subscription(
            GeoPoint,
            'target_position',
            self.target_position_callback,
            10)
        self.trim_state_subscriber = self.create_subscription(
            TrimState,
            'trim_state',
            self.trim_state_callback,
            10)
        self.camera_color_image_subscriber = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.camera_color_image_callback,
            10)
        
        self.camera_mask_image_subscriber = self.create_subscription(
            Image,
            'cv_mask',
            self.camera_mask_image_callback,
            10)
        self.buoy_position_subscriber = self.create_subscription(
            GeoPoint,
            'buoy_position',
            self.buoy_position_callback,
            10)
        self.rudder_angle_subscriber = self.create_subscription(
            Int16,
            'rudder_angle',
            self.rudder_angle_callback,
            10
        )
        self.restart_node_client = self.create_client(RestartNode, 'state_manager/restart_node', callback_group=self.callback_group_state)
        #initial dummy values, for testing
        # self.current_boat_state.latitude = 42.273822
        # self.current_boat_state.longitude = -71.805967
        self.current_boat_state.latitude = 42.276842
        self.current_boat_state.longitude = -71.756035
        self.current_boat_state.current_heading = 0
        self.current_boat_state.track_degrees_true = 0
        self.current_boat_state.track_degrees_magnetic = 0
        self.current_boat_state.speed_knots = 0
        self.current_boat_state.speed_kmh = 0
        self.current_boat_state.rate_of_turn = 0
        self.current_boat_state.true_wind.speed = 0
        self.current_boat_state.true_wind.direction = 270.0
        self.current_boat_state.apparent_wind.speed = 0
        self.current_boat_state.apparent_wind.direction = 0
        self.current_boat_state.pitch = 0
        self.current_boat_state.roll = 0
        self.node_indices = {}
        self.declare_parameter('managed_nodes')
        node_names = self.get_parameter('managed_nodes').get_parameter_value().string_array_value
        self.get_logger().info(f'Active nodes: {node_names}')
        i=0
        for name in node_names:

            #init state
            node_info = boat_state_pb2.NodeInfo()
            node_info.name = name
            node_info.status = boat_state_pb2.NodeStatus.NODE_STATUS_OK
            node_info.info = ""
            self.current_boat_state.node_states.append(node_info)
            self.node_indices[name]=i
            i+=1

            #create lifecycle callback using function generation
            try:
                self.setup_node_subscription(node_name=name)
            except Exception as e:
                trace = traceback.format_exc()
                self.get_logger().fatal(f'Unhandled exception: {e}\n{trace}')

        self.current_boat_state.current_autonomous_mode = boat_state_pb2.AutonomousMode.AUTONOMOUS_MODE_NONE
        # a = boat_state_pb2.Point()
        # a.latitude = 5.1
        # a.longitude = 4.1
        # b=boat_state_pb2.Point()
        # b.latitude = 5.2
        # b.longitude = 4.1
        # self.current_boat_state.current_path.points.append(a)
        # self.current_boat_state.current_path.points.append(b)
        # c = boat_state_pb2.Point()
        # c.latitude = 4.9
        # c.longitude = 3.9
        # d=boat_state_pb2.Point()
        # d.latitude = 4.8
        # d.longitude = 3.9
        # e=boat_state_pb2.Point()
        # e.latitude = 4.7
        # e.longitude = 3.8
        # self.current_boat_state.previous_positions.points.append(c)
        # self.current_boat_state.previous_positions.points.append(d)
        # self.current_boat_state.previous_positions.points.append(e)
        
        self.last_pwm_heartbeat = -1
        self.last_ctrl_heartbeat = -1
        self.last_tt_heartbeat = -1
        try:
            self.create_grpc_server()
        except Exception as e:
            trace = traceback.format_exc()
            self.get_logger().fatal(f'Unhandled exception: {e}\n{trace}')

        self.pwm_heartbeat_subscription = self.create_subscription(
            Empty,
            'heartbeat/pwm_controler',
            self.pwm_controller_heartbeat,
            1)
        
        self.control_system_heartbeat_subscription = self.create_subscription(
            Empty,
            'heartbeat/control_system',
            self.control_system_heartbeat,
            1)
        
        self.trim_tab_heartbeat_subscription = self.create_subscription(
            Empty,
            'heartbeat/trim_tab_comms',
            self.trim_tab_comms_heartbeat,
            1)
        #super().on_configure()
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating...")
        # Start publishers or timers
        #self.ballast_position_publisher.on_activate()
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating...")
        super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up...")
        # Destroy subscribers, publishers, and timers
        self.destroy_lifecycle_publisher(self.pwm_control_publisher)
        self.destroy_lifecycle_publisher(self.ballast_position_publisher)
        self.destroy_lifecycle_publisher(self.trim_tab_control_publisher)
        self.destroy_lifecycle_publisher(self.trim_tab_angle_publisher)
        self.destroy_lifecycle_publisher(self.autonomous_mode_publisher)

        self.destroy_subscription(self.rot_subscription)
        self.destroy_subscription(self.navsat_subscription)
        self.destroy_subscription(self.track_degrees_true_subscription)
        self.destroy_subscription(self.track_degrees_magnetic_subscription)
        self.destroy_subscription(self.speed_knots_subscription)
        self.destroy_subscription(self.speed_kmh_subscription)
        self.destroy_subscription(self.heading_subscription)
        self.destroy_subscription(self.true_wind_subscription)
        self.destroy_subscription(self.apparent_wind_subscription)
        self.destroy_subscription(self.roll_subscription)
        self.destroy_subscription(self.pitch_subscription)
        self.destroy_subscription(self.pwm_heartbeat_subscription)
        self.destroy_subscription(self.control_system_heartbeat_subscription)
        self.destroy_subscription(self.trim_tab_heartbeat_subscription)
        self.destroy_subscription(self.trim_state_subscriber)

        #return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down...")
        # Perform final cleanup if necessary
        return TransitionCallbackReturn.SUCCESS
    
    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().error("Error caught!")
        return super().on_error(state)
     
    #end lifecycle callbacks

    def target_position_callback(self, msg: GeoPoint) -> None:
        self.get_logger().info(f"Sending target point: {msg}")
        self.current_boat_state.current_target_point.latitude = msg.latitude
        self.current_boat_state.current_target_point.longitude = msg.longitude

    def trim_state_callback(self, msg: TrimState) -> None:
        if(msg.state == TrimState.TRIM_STATE_MIN_LIFT):
            self.current_boat_state.current_trim_state = boat_state_pb2.TrimState.TRIM_STATE_MIN_LIFT
        elif(msg.state == TrimState.TRIM_STATE_MAX_LIFT_PORT):
            self.current_boat_state.current_trim_state = boat_state_pb2.TrimState.TRIM_STATE_MAX_LIFT_PORT
        elif(msg.state == TrimState.TRIM_STATE_MAX_LIFT_STARBOARD):
            self.current_boat_state.current_trim_state = boat_state_pb2.TrimState.TRIM_STATE_MAX_LIFT_STARBOARD
        elif(msg.state == TrimState.TRIM_STATE_MAX_DRAG_PORT):
            self.current_boat_state.current_trim_state = boat_state_pb2.TrimState.TRIM_STATE_MAX_DRAG_PORT
        elif(msg.state == TrimState.TRIM_STATE_MAX_DRAG_STARBOARD):
            self.current_boat_state.current_trim_state = boat_state_pb2.TrimState.TRIM_STATE_MAX_DRAG_STARBOARD
        elif(msg.state == TrimState.TRIM_STATE_MANUAL):
            self.current_boat_state.current_trim_state = boat_state_pb2.TrimState.TRIM_STATE_MANUAL
        
    def current_path_callback(self, msg: Path) -> None:
        #self.get_logger().info(f"Updating boat state with new path of length: {len(msg.points)}")
        self.current_boat_state.current_path.ClearField("points")# = command.new_path
        #self.get_logger().info("Cleared old path")
        for geo_point in msg.points:
            point_msg = boat_state_pb2.Point(latitude=geo_point.latitude, longitude = geo_point.longitude)
            self.current_boat_state.current_path.points.append(point_msg)
        #self.get_logger().info("Added new points")

        #self.get_logger().info(f"length of boatState path is: {len(self.current_boat_state.current_path.points)}")

    def restart_lifecycle_node_by_name(self, node_name: str) -> bool:
        # Create a service client to send lifecycle state change requests
        client = self.create_client(ChangeState, f'{node_name}/change_state')

        # Wait for the service to be available
        i=0
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {node_name} lifecycle service...')
            i+=1
            if i > 5:
                return False

        # Deactivate the node
        deactivate_request = ChangeState.Request()
        deactivate_request.transition.id = Transition.TRANSITION_DEACTIVATE
        client.call_async(deactivate_request)
        self.get_logger().info(f'Sent deactivate request to {node_name}')

        # Reactivate the node
        activate_request = ChangeState.Request()
        activate_request.transition.id = Transition.TRANSITION_ACTIVATE
        client.call_async(activate_request)
        self.get_logger().info(f'Sent activate request to {node_name}')
        return True
    
    def create_lifecycle_callback(self, node_name: str) -> Callable[[Any, TransitionEvent], None]:
        def lifecycle_callback(self, msg):
            msg_details = f"Received {node_name} update! State is: {msg.goal_state.id}"
            self.get_logger().info(msg_details)
            self.current_boat_state.node_states[self.node_indices[node_name]].lifecycle_state = get_state(msg.goal_state.id)
        return lifecycle_callback
    
    def setup_node_subscription(self, node_name: str) -> None:
        callback_method_name = f"{node_name}_lifecycle_callback"
        # Dynamically create the callback method
        method = self.create_lifecycle_callback(node_name)
        # Bind the method to the instance, ensuring it receives `self` properly
        bound_method = types.MethodType(method, self)
        # Attach the bound method to the instance
        setattr(self, callback_method_name, bound_method)
        
        # Set up the subscription using the dynamically created and bound callback
        subscription_name = f"{node_name}_lifecycle_subscription"
        topic_name = f"/{node_name}/transition_event"
        subscription = self.create_subscription(
            TransitionEvent,
            topic_name,
            getattr(self, callback_method_name),
            10)
        # Attach the subscription to this class instance
        setattr(self, subscription_name, subscription)

    def rate_of_turn_callback(self, msg: Float64):
        self.current_boat_state.rate_of_turn = msg.data

    def lat_long_callback(self, msg: NavSatFix):
        #self.get_logger().info(f"Got latlong: {msg.latitude}, {msg.longitude}")
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
        self.current_boat_state.true_wind.speed = msg.speed
        self.current_boat_state.true_wind.direction = msg.direction

    def apparent_wind_callback(self, msg: Wind):
        self.current_boat_state.apparent_wind.speed = msg.speed
        self.current_boat_state.apparent_wind.direction = msg.direction

    def roll_callback(self, msg: Float64):
        self.current_boat_state.roll = msg.data
    
    def pitch_callback(self, msg: Float64):
        self.current_boat_state.pitch = msg.data
    
    def set_current_image(self, msg: Image):
        current_time = time.time()
        if(current_time>self.last_camera_frame_time+0.1):
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            self.last_camera_frame_shape = frame.shape
            self.last_camera_frame = encode_frame(frame)
            self.last_camera_frame_time = current_time

    def camera_color_image_callback(self, msg: Image):
        # if(self.do_video_encode == False):
        #     return
        if(self.current_video_source != 0):
            return
        self.set_current_image(msg)

    def camera_mask_image_callback(self, msg: Image):
        # if(self.do_video_encode == False):
        #     return
        if(self.current_video_source != 1):
            return
        self.set_current_image(msg)

    def buoy_position_callback(self, msg: GeoPoint):
        self.current_boat_state.ClearField("buoy_positions")# = command.new_path
        #self.get_logger().info("Cleared old path")
        point_msg = boat_state_pb2.Point(latitude=msg.latitude, longitude = msg.longitude)
        self.current_boat_state.buoy_positions.append(point_msg)

    def rudder_angle_callback(self, msg: Int16):
        self.current_boat_state.rudder_position = msg.data

    #new server code
    def create_grpc_server(self): 
        self.get_logger().info("Creating gRPC server")
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        control_pb2_grpc.add_ExecuteRudderCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteTrimTabCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteBallastCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteAutonomousModeCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteSetWaypointsCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteAddWaypointCommandServiceServicer_to_server(self, self.grpc_server)
        control_pb2_grpc.add_ExecuteMarkBuoyCommandServiceServicer_to_server(self, self.grpc_server)
        boat_state_pb2_rpc.add_SendBoatStateServiceServicer_to_server(self, self.grpc_server)
        boat_state_pb2_rpc.add_GetMapServiceServicer_to_server(self, self.grpc_server)
        boat_state_pb2_rpc.add_StreamBoatStateServiceServicer_to_server(self, self.grpc_server)
        node_restart_pb2_grpc.add_RestartNodeServiceServicer_to_server(self, self.grpc_server)
        video_pb2_grpc.add_VideoStreamerServicer_to_server(self, self.grpc_server)

        #connect_pb2_grpc.add_ConnectToBoatServiceServicer_to_server(self, self.grpc_server)
        self.grpc_server.add_insecure_port('[::]:50051')
        self.grpc_server.start()

    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def StreamVideo(self, command: video_pb2.VideoRequest, context):
        #self.do_video_encode = True
        rate = self.create_rate(10)
        if(command.videoSource == 'COLOR'):
            self.current_video_source = 0
        elif(command.videoSource == 'MASK'):
            self.current_video_source = 1

        try:
            while context.is_active():
                yield video_pb2.VideoFrame(data=self.last_camera_frame, width=self.last_camera_frame_shape[1], height=self.last_camera_frame_shape[0], timestamp=int(time.time()))
                rate.sleep()
        finally:
            if not context.is_active():
                #self.do_video_encode = False
                self.get_logger().info("Video stream was cancelled or client disconnected.")

    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteMarkBuoyCommand(self, command: control_pb2.MarkBuoyCommand, context):
        self.get_logger().info("Received mark buoy command")

    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteRudderCommand(self, command: control_pb2.RudderCommand, context):
        #center = 75 degrees, full right=40, full left = 113
        response = control_pb2.ControlResponse()
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_SUCCESS
        #rudder commands are inverted radians, map to degrees and invert
        degrees = command.rudder_control_value*(180/math.pi)*-1
        self.get_logger().info(f"degrees: {degrees}")
        
        msg = Int16()
        msg.data = int(degrees)
        self.rudder_control_publisher.publish(msg)
        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteTrimTabCommand(self, command: control_pb2.TrimTabCommand, context):
        response = control_pb2.ControlResponse()
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_SUCCESS
        state_msg = Int8()
        state_msg.data = 5
        angle_msg = Int16()
        angle_msg.data = int((command.trimtab_control_value+math.pi/2)*180/math.pi)
        self.trim_tab_control_publisher.publish(state_msg)
        self.trim_tab_angle_publisher.publish(angle_msg)
        #self.get_logger().info("Publishing trimtab command")
        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteBallastCommand(self, command: control_pb2.BallastCommand, context):
        response = control_pb2.ControlResponse()
        self.get_logger().info("Publishing ballast command")
        position_msg = Float64()
        position_msg.data = command.ballast_control_value
        self.ballast_position_publisher.publish(position_msg)

        # current_target = 80 + ((110 - 80) / (1.0 - -1.0)) * (command.ballast_control_value - -1.0)
        # ballast_json = {"channel": "12", "angle": current_target}
        # self.pwm_control_publisher.publish(make_json_string(ballast_json))

        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_SUCCESS
        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteAutonomousModeCommand(self, command: control_pb2.AutonomousModeCommand, context):
        self.get_logger().info(f"Received autonomous mode command: {command.autonomous_mode}")
        msg = AutonomousMode()
        if command.autonomous_mode == boat_state_pb2.AutonomousMode.AUTONOMOUS_MODE_NONE:
            msg.mode = AutonomousMode.AUTONOMOUS_MODE_NONE
        elif command.autonomous_mode == boat_state_pb2.AutonomousMode.AUTONOMOUS_MODE_BALLAST:
            msg.mode = AutonomousMode.AUTONOMOUS_MODE_BALLAST
        elif command.autonomous_mode == boat_state_pb2.AutonomousMode.AUTONOMOUS_MODE_TRIMTAB:
            msg.mode = AutonomousMode.AUTONOMOUS_MODE_TRIMTAB
        elif command.autonomous_mode == boat_state_pb2.AutonomousMode.AUTONOMOUS_MODE_FULL:
            msg.mode = AutonomousMode.AUTONOMOUS_MODE_FULL
        self.autonomous_mode_publisher.publish(msg)
        self.current_boat_state.autonomous_mode = command.autonomous_mode
        response = control_pb2.ControlResponse()
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_SUCCESS
        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteSetWaypointsCommand(self, command: control_pb2.SetWaypointsCommand, context):
        self.get_logger().info("Received waypoints command")
        response = control_pb2.ControlResponse()
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_SUCCESS
        self.current_boat_state.current_waypoints.ClearField("waypoints")# = command.new_path
        self.current_boat_state.current_waypoints.waypoints.extend(command.new_waypoints.waypoints)
        self.get_logger().info(f"Received waypoints with {len(command.new_waypoints.waypoints)} points: ")
        waypoints = WaypointPath()
        for waypoint in command.new_waypoints.waypoints:
            self.get_logger().info(str(waypoint.point.latitude)+" : "+str(waypoint.point.longitude))
            fix = Waypoint()
            fix.point.latitude = waypoint.point.latitude
            fix.point.longitude = waypoint.point.longitude
            if(waypoint.type == boat_state_pb2.WaypointType.WAYPOINT_TYPE_INTERSECT):
                fix.type = Waypoint.WAYPOINT_TYPE_INTERSECT
            elif(waypoint.type == boat_state_pb2.WaypointType.WAYPOINT_TYPE_CIRCLE_RIGHT):
                fix.type = Waypoint.WAYPOINT_TYPE_CIRCLE_RIGHT
            elif(waypoint.type == boat_state_pb2.WaypointType.WAYPOINT_TYPE_CIRCLE_LEFT):
                fix.type = Waypoint.WAYPOINT_TYPE_CIRCLE_LEFT
            waypoints.waypoints.append(fix)

        self.get_logger().info("Publishing waypoints")
        self.waypoints_publisher.publish(waypoints)

        return response
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def ExecuteAddWaypointCommand(self, command: control_pb2.AddWaypointCommand, context):
        self.get_logger().info("Received add single waypoint command")
        response = control_pb2.ControlResponse()
        response.execution_status = control_pb2.ControlExecutionStatus.CONTROL_EXECUTION_SUCCESS
        self.current_boat_state.current_waypoints.waypoints.append(command.new_waypoint)
        waypoint = Waypoint()
        waypoint.point.latitude = command.new_waypoint.point.latitude
        waypoint.point.longitude = command.new_waypoint.point.longitude
        if(command.new_waypoint.type == boat_state_pb2.WaypointType.WAYPOINT_TYPE_INTERSECT):
            waypoint.type = Waypoint.WAYPOINT_TYPE_INTERSECT
        elif(command.new_waypoint.type == boat_state_pb2.WaypointType.WAYPOINT_TYPE_CIRCLE_RIGHT):
            waypoint.type = Waypoint.WAYPOINT_TYPE_CIRCLE_RIGHT
        elif(command.new_waypoint.type == boat_state_pb2.WaypointType.WAYPOINT_TYPE_CIRCLE_LEFT):
            waypoint.type = Waypoint.WAYPOINT_TYPE_CIRCLE_LEFT

        self.get_logger().info("Publishing single waypoint")
        self.single_waypoint_publisher.publish(waypoint)
        return response

    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def GetMap(self, command: boat_state_pb2.MapRequest, context):
        self.get_logger().info("Received GetMap request")
        _, buffer = cv2.imencode('.png', self.map_image)
        #self.get_logger().info(f"Image buffer: {buffer}")
        response = boat_state_pb2.MapResponse()
        response.image_data = buffer.tobytes()
        response.north = self.bbox['north']
        response.south = self.bbox['south']
        response.east = self.bbox['east']
        response.west = self.bbox['west']
        return response


    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def SendBoatState(self, command: boat_state_pb2.BoatStateRequest, context):
        return self.current_boat_state
    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def StreamBoatState(self, command: boat_state_pb2.BoatStateRequest, context):
        rate = self.create_rate(1)
        try:
            while context.is_active():
                yield self.current_boat_state
                rate.sleep()
        finally:
            if not context.is_active():
                self.get_logger().info("Boat state stream was cancelled or client disconnected.")

    
    #gRPC function, do not rename unless you change proto defs and recompile gRPC files
    def RestartNode(self, command: node_restart_pb2.RestartNodeRequest, context):
        self.get_logger().info("Received restart command for: "+command.node_name)
        # restart_node_request = RestartNode.Request()
        # restart_node_request.node_name = command.node_name
        # if(self.restart_node_client.wait_for_service(3) is False):
        #     self.get_logger().error("Client service not available for state manager!")
        #     response.success = False
        #     return response
        # future = self.restart_node_client.call_async(restart_node_request)
        # self.get_logger().info("About to spin")
        # rclpy.spin_until_future_complete(self, future)
        # self.get_logger().info("completed command")
        result = self.restart_lifecycle_node_by_name(command.node_name)
        response = node_restart_pb2.RestartNodeResponse()
        response.success = result
        self.get_logger().info("Restart node: "+str(result))
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

    # Use the SingleThreadedExecutor to spin the node.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(network_comms)

    try:
        # Spin the node to execute callbacks
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        trace = traceback.format_exc()
        network_comms.get_logger().fatal(f'Unhandled exception: {e}\n{trace}')
    finally:
        # Shutdown and cleanup the node
        executor.shutdown()
        network_comms.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()