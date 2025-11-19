#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from xela_server_ros2.msg import SensStream, SensorFull

import numpy as np


############################################################
class PalmSensorRecorder(Node):

    def __init__(self):
        super().__init__('palm_sensor_recorder')

        # subscribe to finger sensors
        self.palm_sub = self.create_subscription(SensStream, 'xPalmTopic', self.fingers_cb, 10)

        self.current_mean = None
        self.count = 0

        self.max_count = 20000

        np.set_printoptions(precision=3)


    def fingers_cb(self, msg: SensStream):
        # Palm sensor pos ordering:
        # 1 - top right
        # 2 - top left
        # 3 - bottom left
        sensors = msg.sensors
        taxels, forces = extract_palm_sensor_data(sensors[SENSOR_IDX - 1])    # shape = (4, 6, 3)

        if self.current_mean is None:
            self.current_mean = taxels
        else:
            self.current_mean = (self.current_mean * self.count + taxels) / (self.count + 1)
        self.count += 1

        if self.count % 2000 == 0:
            print(f"After {self.count} messages, mean tactile data for sensor {SENSOR_IDX}:")
            print(self.current_mean)

        if self.count >= self.max_count:
            save_dir = "/home/mpan31415/ros2_ws/src/TactileManip/hand_interface/scripts/calib/taxel_mean/"
            filename = save_dir + f"palm_sensor_{SENSOR_IDX}_mean.npy"
            self.save_to_file(filename)
            rclpy.shutdown()


    def save_to_file(self, filename: str):
        if self.current_mean is not None:
            np.save(filename, self.current_mean)
            print("=" * 1000)
            print("\n")
            print(f"Saved mean tactile data to {filename}")
            print("\n")
            print("=" * 1000)
        else:
            print("No data to save.")



def extract_palm_sensor_data(sensor: SensorFull):
    taxel_msgs = sensor.taxels
    force_msgs = sensor.forces
    # convert to numpy arrays, shape = (4, 6, 3)
    taxels = np.array([[taxel_msg.x, taxel_msg.y, taxel_msg.z] for taxel_msg in taxel_msgs]).reshape((4, 6, 3))
    forces = np.array([[force_msg.x, force_msg.y, force_msg.z] for force_msg in force_msgs]).reshape((4, 6, 3))
    return taxels, forces


############################################################
if __name__ == '__main__':
    
    # get argument from command line for SENSOR_IDX
    import sys
    if len(sys.argv) > 1:
        SENSOR_IDX = int(sys.argv[1])
    else:
        SENSOR_IDX = 1
    print("Using SENSOR_IDX =", SENSOR_IDX)
    
    rclpy.init(args=None)

    node = PalmSensorRecorder()
    rclpy.spin(node)
