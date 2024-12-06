import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ServerGoalHandle
from sailbot_msgs.action import CrossWind
from std_msgs.msg import Float64, Int16, Empty
import math, time

class TackingActionServer(Node):
    def __init__(self):
        super().__init__('tacking_action_server')
        self._action_server = ActionServer(
            self,
            CrossWind,
            'perform_tack',
            self.execute_callback
        )

        self.heading = 0.0
        self.wind_dir = 0.0
        self.subscription = self.create_subscription(Float64, 'heading', self.heading_callback, 10)
        self.wind_subscription = self.create_subscription(Float64, 'wind_direction', self.wind_callback, 10)

        # Publisher to command rudder angle
        self.rudder_angle_pub = self.create_publisher(Int16, 'rudder_angle', 10)

    def heading_callback(self, msg: Float64):
        self.heading = msg.data

    def wind_callback(self, msg: Float64):
        self.wind_dir = msg.data

    def execute_callback(self, goal_handle: ServerGoalHandle):
        self.get_logger().info('Received a tack/jibe goal')

        wind_direction_deg = goal_handle.request.wind_direction_deg
        timeout = goal_handle.request.timeout
        direction = goal_handle.request.direction
        go_long_way = goal_handle.request.go_long_way

        start_time = time.time()

        # Determine initial side relative to wind
        initial_error = self.compute_heading_error(self.heading, wind_direction_deg)
        starting_side = 1 if initial_error > 0 else -1

        # Initiate maneuver - if normal tack or jibe, apply rudder accordingly
        rudder_cmd = Int16()
        # If go_long_way is True, we might choose a different initial rudder angle or the same
        # For simplicity, use the same logic:
        rudder_cmd.data = 30 * direction
        self.rudder_angle_pub.publish(rudder_cmd)

        success = False
        rate = self.create_rate(1)  # 1 Hz
        required_error = 20.0

        # Define success conditions based on go_long_way
        # Normal tack logic:
        # - starting_side = -1 (left): success if heading_error > +20
        # - starting_side = +1 (right): success if heading_error < -20

        # Long way (jibe) logic:
        # - starting_side = -1: success if heading_error < -20 (opposite side around the other way)
        # - starting_side = +1: success if heading_error > +20

        while rclpy.ok():
            elapsed = time.time() - start_time
            heading_error = self.compute_heading_error(self.heading, wind_direction_deg)

            # Feedback
            feedback_msg = CrossWind.Feedback()
            feedback_msg.current_heading = self.heading
            feedback_msg.heading_error = heading_error
            feedback_msg.elapsed_time = float(elapsed)
            goal_handle.publish_feedback(feedback_msg)

            if not go_long_way:
                # Normal tack success conditions
                if starting_side == -1 and heading_error > required_error:
                    success = True
                    break
                elif starting_side == 1 and heading_error < -required_error:
                    success = True
                    break
            else:
                # Jibe (long way) success conditions
                if starting_side == -1 and heading_error < -required_error:
                    success = True
                    break
                elif starting_side == 1 and heading_error > required_error:
                    success = True
                    break

            if elapsed > timeout:
                break

            rate.sleep()

        # Stop maneuvering
        rudder_cmd.data = 0
        self.rudder_angle_pub.publish(rudder_cmd)

        result = CrossWind.Result()
        if success:
            if go_long_way:
                result.message = "Successfully completed jibe (long way around)."
            else:
                result.message = "Successfully completed tack."
            result.success = True
            self.get_logger().info(result.message)
        else:
            result.success = False
            if go_long_way:
                result.message = "Jibe failed to complete in time."
            else:
                result.message = "Tack failed to complete in time."
            self.get_logger().warn(result.message)

        return result

    def compute_heading_error(self, heading, wind_dir):
        # Compute how far off we are from crossing the wind direction
        error = (wind_dir - heading + 180) % 360 - 180
        return error

def main(args=None):
    rclpy.init(args=args)
    node = TackingActionServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()