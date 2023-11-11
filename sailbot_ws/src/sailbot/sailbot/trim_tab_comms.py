#!/usr/bin/env python3
import struct
import asyncio
import os

from bleak import BleakClient
from bleak.uuids import uuid16_dict
import rclpy
from typing import Optional
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.timer import Timer
from rclpy.subscription import Subscription
from enum import Enum

from std_msgs.msg import Int8, Int16, Float32, Empty


ble_address = "30:c6:f7:02:70:56"  # Trim Tab controller BLE address
uuid16_dict = {v: k for k, v in uuid16_dict.items()}

# Characteristic UUIDs
APPARENT_WIND_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Apparent Wind Direction")
)
BATTERY_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Battery Level")
)
TAB_STATE_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("LN Feature")
)
TAB_ANGLE_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("LN Control Point")
)


class TRIM_STATE(int, Enum):
    MAX_LIFT_PORT: int = 0
    MAX_LIFT_STBD: int = 1
    MAX_DRAG_PORT: int = 2
    MAX_DRAG_STBD: int = 3
    MIN_LIFT: int = 4
    MANUAL: int = 5


# Local variables
state = TRIM_STATE.MIN_LIFT
angle = 0
wind_dir = 0.0
battery_level = 100


class TrimTabComms(LifecycleNode):
    def __init__(self):
        super(TrimTabComms, self).__init__('trim_tab_comms')

        self.tt_telemetry_publisher: Optional[Publisher]
        self.tt_battery_publisher: Optional[Publisher]
        self.tt_control_subscriber: Optional[Subscription]
        self.tt_angle_subscriber: Optional[Subscription]
        self.timer_pub: Optional[Publisher]

        self.heartbeat_timer: Optional[Timer]
        self.timer: Optional[Timer]

    #lifecycle node callbacks
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("In configure")
        self.tt_telemetry_publisher = self.create_lifecycle_publisher(Float32, 'tt_telemetry', 10)  # Wind direction
        self.tt_battery_publisher = self.create_lifecycle_publisher(Int8, 'tt_battery', 10)  # Battery level
        self.tt_control_subscriber = self.create_subscription(Int8, 'tt_control', self.listener_callback, 10)  # Trim tab state
        self.tt_angle_subscriber = self.create_subscription(Int16, 'tt_angle', self.angle_callback, 10)

        self.timer_pub = self.create_lifecycle_publisher(
        Empty, '/heartbeat/trim_tab_comms', 1)
        self.heartbeat_timer = self.create_timer(0.5, self.heartbeat_timer_callback)

        # Set up timer for retrieving variables over BLE
        timer_period = 0.5  # Fetch data every 0.5 seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        #restart bluetooth- necessary to kill hanging connections if ros node crashes and restarts
        os.system("rfkill block bluetooth")
        os.system("rfkill unblock bluetooth")

        # Set up BLE client and attempt to connect
        self.client = BleakClient(ble_address)
        self.expect_disconnect = False
        self.client.set_disconnected_callback(self.lost_connection)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ble_connect())
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating...")
        # Start publishers or timers
        return super().on_activate(state)


    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating...")
        super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up...")
        #disconnect from and reset bluetooth
        self.disconnect()
        os.system("rfkill block bluetooth")
        os.system("rfkill unblock bluetooth")
        # Destroy subscribers, publishers, and timers
        self.destroy_lifecycle_publisher(self.tt_telemetry_publisher)
        self.destroy_lifecycle_publisher(self.tt_battery_publisher)
        self.destroy_lifecycle_publisher(self.timer_pub)
        self.destroy_subscription(self.tt_control_subscriber)
        self.destroy_subscription(self.tt_angle_subscriber)
        self.destroy_timer(self.heartbeat_timer)
        self.destroy_timer(self.timer)

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down...")
        # Perform final cleanup if necessary
        return TransitionCallbackReturn.SUCCESS
    
    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Error caught!")
        return super().on_error(state)
    
    #end callbacks

    def lost_connection(self, client):
        """ Attempt to reconnect if connection is lost unexpectedly """
        if self.expect_disconnect:
            return None

        self.get_logger().info("Lost connection: Attempting to reconnect...")
        loop = asyncio.get_event_loop()
        loop.create_task(self._ble_connect())

    def disconnect(self):
        """ Disconnect from the trim tab controller """
        self.expect_disconnect = True
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ble_disconnect())

    async def _ble_connect(self):
        """ Attempt to establish a connection over BLE [Asynchronous] """
        attempts = 0
        connected = False
        while not connected and attempts < 5:
            try:
                await self.client.connect()
                self.get_logger().info("Connected")
                connected = True
            except Exception as e:
                self.get_logger().debug(str(e))
                attempts += 1

        if not connected:
            self.get_logger().error("Failed to connect to trim tab controller. Is it on?")

    async def _ble_disconnect(self):
        """ Attempt to disconnect from BLE [Asynchronous] """
        try:
            await self.client.disconnect()
            self.get_logger().info("Disconnected")
        except Exception as e:
            self.get_logger().error(str(e))

    async def ble_read(self):
        """ Retrieve characteristic values from the trim tab controller [Asynchronous] """
        try:
            global wind_dir
            global battery_level
            wind_dir = struct.unpack('f', await self.client.read_gatt_char(APPARENT_WIND_UUID))[0]
            battery_data = await self.client.read_gatt_char(BATTERY_UUID)
            battery_level = int(battery_data[0])
        except Exception as e:
            self.get_logger().error("read failed: "+str(e))

    async def ble_write(self):
        """ Update the state of the trim tab controller [Asynchronous] """
        try:
            global state
            global angle
            await self.client.write_gatt_char(TAB_STATE_UUID, state.to_bytes(1, 'little'))
            if state == TRIM_STATE.MANUAL:
                await self.client.write_gatt_char(TAB_ANGLE_UUID, angle.to_bytes(1, 'little'))
        except Exception as e:
            self.get_logger().error("Write failed: "+str(e))

    def timer_callback(self):
        """
        Handles timer event. Retrieves the new values from the trim tab controller and publishes them over ROS.
        """
        self.get_logger().debug("Attempting to read BLE variables")
        global battery_level
        global wind_dir
        
        # Read current values from trim tab controller
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ble_read())

        # Publish ROS messages
        telemetry_msg = Float32()
        battery_msg = Int8()
        telemetry_msg.data = float(wind_dir)
        battery_msg.data = battery_level
        self.tt_telemetry_publisher.publish(telemetry_msg)
        self.tt_battery_publisher.publish(battery_msg)

    def listener_callback(self, msg):
        """ Subscription handler for trim tab state commands """
        self.get_logger().debug("Sending command to trim tab controller...")
        global state
        state = msg.data
        
        # Send new state to trim tab controller
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ble_write())
    
    def angle_callback(self, msg):
        """ Subscription handler for trim tab angle updates (Manual mode) """
        self.get_logger().debug("Updating manual trim tab angle")
        global angle
        angle = msg.data

        # Send new angle to trim tab controller
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ble_write())
    
    def heartbeat_timer_callback(self):
      #self.get_logger().info("Publishing timer")
      self.timer_pub.publish(Empty())


def main(args=None):
    rclpy.init(args=args)

    tt_comms = TrimTabComms()

    try:
        rclpy.spin(tt_comms)

        # Clean up when we are done
        tt_comms.get_logger().info("Node exiting")
        tt_comms.disconnect()
        tt_comms.destroy_node()
        rclpy.shutdown()
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        tt_comms.disconnect()
        raise


if __name__ == '__main__':
    main()
