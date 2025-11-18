#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time

class AllegroHandCommand(Node):
    def __init__(self):
        super().__init__('allegro_hand_command')

        # Publisher
        self.pub = self.create_publisher(JointState, '/allegroHand_0/joint_cmd', 10)

        # Timer for periodic publishing
        self.command_timer = self.create_timer(0.05, self.command_cb)  # 20 Hz

        # subscribe to joint states
        self.subscriber = self.create_subscription(
            JointState,
            "/allegroHand_0/joint_states",
            self.joint_state_callback,
            10
        )

        self.last_print_time = time.time()

        # Joint names (from your joint_states topic)
        self.joint_names = [
            'joint_0', 'joint_1', 'joint_2', 'joint_3',
            'joint_4', 'joint_5', 'joint_6', 'joint_7',
            'joint_8', 'joint_9', 'joint_10', 'joint_11',
            'joint_12', 'joint_13', 'joint_14', 'joint_15'
        ]

        # Target joint positions
        self.positions = [
            0.0, 0.85, 0.8, 0.2,
            0.0, 0.85, 0.8, 0.2,
            0.0, 0.85, 0.8, 0.2,
            1.3, 0.6, 0.1, 0.7
        ]

        self.current_positions = None

    def command_cb(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.positions
        self.pub.publish(msg)

    def joint_state_callback(self, msg: JointState):
        # Only proceed if the message contains positions
        if not msg.position or len(msg.position) < len(self.positions):
            return

        self.current_positions = msg.position

        # Print deviation every 0.5 seconds
        now = time.time()
        if now - self.last_print_time >= 0.5:
            deviation_sum = sum(abs(a - b) for a, b in zip(self.positions, self.current_positions))
            self.get_logger().info(f"Total joint deviation: {deviation_sum:.4f}")
            self.last_print_time = now


def main(args=None):
    rclpy.init(args=args)
    node = AllegroHandCommand()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
