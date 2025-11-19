#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from xela_server_ros2.msg import SensStream, SensorFull

import numpy as np
import matplotlib.pyplot as plt


############################################################
class TactileListener(Node):

    def __init__(self):
        super().__init__('tactile_listener')

        # subscribe to palm & finger sensors
        self.palm_sub = self.create_subscription(SensStream, 'xPalmTopic', self.palm_cb, 10)
        # self.fingers_sub = self.create_subscription(SensStream, 'xFingersTopic', self.fingers_cb, 10)

        # tactile data buffer
        self.palm_tactile_data = np.zeros((3, 4, 6, 3))
        self.finger_tactile_data = np.zeros((15, 4, 4, 3))

        # palm data plotter
        plot_update_freq = 20.0    # Hz
        self.timer = self.create_timer(1 / plot_update_freq, self.update_palm_plot)

        # initialize plots
        self.init_palm_plot()


    # TODO:
    # - subsample tactile data (e.g. 100 Hz or 50 Hz)
        
        
    def palm_cb(self, msg: SensStream):
        # Palm sensor pos ordering:
        # 1 - top right
        # 2 - top left
        # 3 - bottom left
        sensors = msg.sensors
        for sensor_idx in range(1, 4):
            pos, taxels, forces = extract_palm_sensor_data(sensors[sensor_idx - 1])
            self.palm_tactile_data[sensor_idx - 1, :, :, :] = forces

    def fingers_cb(self, msg: SensStream):
        pass
            
    
    def init_palm_plot(self):
        plt.ion()
        self.palm_fig, self.palm_axes = plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True)

        # define the 3 axes
        ax_top_right   = self.palm_axes[0,1]
        ax_top_left    = self.palm_axes[0,0]
        ax_bottom_left = self.palm_axes[1,0]
        used_axes = [ax_top_right, ax_top_left, ax_bottom_left]

        # initialize 3 heatmaps; each must be a 4×6 array
        self.palm_heatmaps = [np.zeros((4, 6)) for _ in range(3)]

        self.palm_images = []
        for ax, d in zip(used_axes, self.palm_heatmaps):
            img = ax.imshow(d, vmin=0.0, vmax=5.0, cmap='viridis')
            self.palm_images.append(img)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused bottom-right axis
        self.palm_axes[1,1].set_visible(False)
        # Add shared colorbar on the right
        self.palm_fig.colorbar(self.palm_images[0], ax=used_axes, location='right', pad=0.03)
        plt.show()
        

    def update_palm_plot(self):
        tactile_norms = np.linalg.norm(self.palm_tactile_data, axis=3)    # shape (3, 4, 6)

        # update plot data
        for i in range(3):
            self.palm_heatmaps[i] = tactile_norms[i]
        for img, d in zip(self.palm_images, self.palm_heatmaps):
            img.set_data(d)

        # redraw only the changed elements
        self.palm_fig.canvas.draw_idle()
        plt.pause(0.001)


def extract_palm_sensor_data(sensor: SensorFull):
    pos = sensor.sensor_pos
    taxel_msgs = sensor.taxels
    force_msgs = sensor.forces
    # convert to numpy arrays, shape = (4, 6, 3)
    taxels = np.array([[taxel_msg.x, taxel_msg.y, taxel_msg.z] for taxel_msg in taxel_msgs]).reshape((4, 6, 3))
    forces = np.array([[force_msg.x, force_msg.y, force_msg.z] for force_msg in force_msgs]).reshape((4, 6, 3))
    # if 2nd or 3rd sensor, rotate data by 180 degrees to match physical orientation
    if pos in [2, 3]:
        taxels = np.rot90(taxels, 2, axes=(0, 1))
        forces = np.rot90(forces, 2, axes=(0, 1))
    return pos, taxels, forces


############################################################
if __name__ == '__main__':
    
    rclpy.init(args=None)

    node = TactileListener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()