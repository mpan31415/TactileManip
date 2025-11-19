#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from xela_server_ros2.msg import SensStream, SensorFull

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


############################################################
class TactileListener(Node):

    def __init__(self):
        super().__init__('tactile_listener')

        # subscribe to palm & finger sensors
        self.palm_sub = self.create_subscription(SensStream, 'xPalmTopic', self.palm_cb, 10)
        self.fingers_sub = self.create_subscription(SensStream, 'xFingersTopic', self.fingers_cb, 10)

        # tactile data buffer
        self.palm_tactile_data = np.zeros((3, 4, 6, 3))
        self.finger_tactile_data = np.zeros((15, 4, 4, 3))

        # palm data plotter
        plot_update_freq = 10.0    # Hz
        # self.timer = self.create_timer(1 / plot_update_freq, self.update_tactile_plot)

        # initialize plots
        # self.init_tactile_plot()


    # TODO:
    # - subsample tactile data (e.g. 100 Hz or 50 Hz)
        
        
    def palm_cb(self, msg: SensStream):
        # Palm sensor pos ordering:
        # 1 - top right
        # 2 - top left
        # 3 - bottom left
        sensors = msg.sensors
        for i in range(3):
            taxels, forces = extract_palm_sensor_data(sensors[i])
            self.palm_tactile_data[i, :, :, :] = forces

    def fingers_cb(self, msg: SensStream):
        # Fingers sensor pos ordering:
        # [1, 2, 3] - thumb (tip to base)
        # [4, 5, 6, 7] - index (tip to base)
        # [8, 9, 10, 11] - middle (tip to base)
        # [12, 13, 14, 15] - ring (tip to base)
        sensors = msg.sensors
        for i in range(15):
            taxels = extract_finger_sensor_data(sensors[i])
            self.finger_tactile_data[i, :, :, :] = taxels

        min_val = np.min(self.finger_tactile_data)
        max_val = np.max(self.finger_tactile_data)
        print(f"Finger tactile data range: {min_val:.2f} to {max_val:.2f}")
            
    
    def init_tactile_plot(self):
        plt.ion()
        self.tactile_fig = plt.figure(figsize=(12, 8), constrained_layout=True)

        # -----------------------------
        # GridSpec layout (hand shape)
        # -----------------------------
        gs = GridSpec(12, 8, figure=self.tactile_fig)  # flexible hand layout

        # ---- Thumb: 3 heatmaps ----
        thumb_ax1 = self.tactile_fig.add_subplot(gs[6:8, 6:8])
        thumb_ax2 = self.tactile_fig.add_subplot(gs[8:10, 6:8])
        thumb_ax3 = self.tactile_fig.add_subplot(gs[10:12, 6:8])
        self.thumb_axes = [thumb_ax1, thumb_ax2, thumb_ax3]

        # store finger axes in a dict
        finger_positions = ["index", "middle", "ring"]
        self.finger_axes = {k: [] for k in finger_positions}

        # ---- Index: 4 heatmaps ----
        index_ax1 = self.tactile_fig.add_subplot(gs[0:2, 4:6])
        index_ax2 = self.tactile_fig.add_subplot(gs[2:4, 4:6])
        index_ax3 = self.tactile_fig.add_subplot(gs[4:6, 4:6])
        index_ax4 = self.tactile_fig.add_subplot(gs[6:8, 4:6])
        self.finger_axes["index"] = [index_ax1, index_ax2, index_ax3, index_ax4]

        # ---- Middle: 4 heatmaps ----
        middle_ax1 = self.tactile_fig.add_subplot(gs[0:2, 2:4])
        middle_ax2 = self.tactile_fig.add_subplot(gs[2:4, 2:4])
        middle_ax3 = self.tactile_fig.add_subplot(gs[4:6, 2:4])
        middle_ax4 = self.tactile_fig.add_subplot(gs[6:8, 2:4])
        self.finger_axes["middle"] = [middle_ax1, middle_ax2, middle_ax3, middle_ax4]

        # ---- Ring: 4 heatmaps ----
        ring_ax1 = self.tactile_fig.add_subplot(gs[0:2, 0:2])
        ring_ax2 = self.tactile_fig.add_subplot(gs[2:4, 0:2])
        ring_ax3 = self.tactile_fig.add_subplot(gs[4:6, 0:2])
        ring_ax4 = self.tactile_fig.add_subplot(gs[6:8, 0:2])
        self.finger_axes["ring"] = [ring_ax1, ring_ax2, ring_ax3, ring_ax4]

        # ---- Palm: 3 heatmaps ----
        palm_ax1 = self.tactile_fig.add_subplot(gs[8:10, 3:6])
        palm_ax2 = self.tactile_fig.add_subplot(gs[8:10, 0:3])
        palm_ax3 = self.tactile_fig.add_subplot(gs[10:12, 0:3])
        self.palm_axes = [palm_ax1, palm_ax2, palm_ax3]

        # -----------------------------
        # Initialize heatmaps
        # -----------------------------
        self.palm_heatmaps = [np.zeros((4, 6)) for _ in range(3)]
        self.thumb_heatmaps = [np.zeros((4, 4)) for _ in range(3)]
        self.index_heatmaps = [np.zeros((4, 4)) for _ in range(4)]
        self.middle_heatmaps = [np.zeros((4, 4)) for _ in range(4)]
        self.ring_heatmaps = [np.zeros((4, 4)) for _ in range(4)]

        self.palm_images = []
        for ax, d in zip(self.palm_axes, self.palm_heatmaps):
            img = ax.imshow(d, vmin=0.0, vmax=5.0, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
            self.palm_images.append(img)

        self.thumb_images = []
        for ax, d in zip(self.thumb_axes, self.thumb_heatmaps):
            img = ax.imshow(d, vmin=50000, vmax=70000, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
            self.thumb_images.append(img)

        self.index_images = []
        for ax, d in zip(self.finger_axes["index"], self.index_heatmaps):
            img = ax.imshow(d, vmin=50000, vmax=70000, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
            self.index_images.append(img)

        self.middle_images = []
        for ax, d in zip(self.finger_axes["middle"], self.middle_heatmaps):
            img = ax.imshow(d, vmin=50000, vmax=70000, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
            self.middle_images.append(img)

        self.ring_images = []
        for ax, d in zip(self.finger_axes["ring"], self.ring_heatmaps):
            img = ax.imshow(d, vmin=50000, vmax=70000, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
            self.ring_images.append(img)

        plt.show()


    def update_tactile_plot(self):
        # data to plot
        palm_norms = np.linalg.norm(self.palm_tactile_data, axis=3)
        thumb_norms = np.linalg.norm(self.finger_tactile_data[0:3, :, :, :], axis=3)
        index_norms = np.linalg.norm(self.finger_tactile_data[3:7, :, :, :], axis=3)
        middle_norms = np.linalg.norm(self.finger_tactile_data[7:11, :, :, :], axis=3)
        ring_norms = np.linalg.norm(self.finger_tactile_data[11:15, :, :, :], axis=3)

        # update palm
        for i in range(3):
            self.palm_heatmaps[i] = palm_norms[i]
        for img, d in zip(self.palm_images, self.palm_heatmaps):
            img.set_data(d)

        # update thumb
        for i in range(3):
            self.thumb_heatmaps[i] = thumb_norms[i]
        for img, d in zip(self.thumb_images, self.thumb_heatmaps):
            img.set_data(d)

        # update index
        for i in range(4):
            self.index_heatmaps[i] = index_norms[i]
        for img, d in zip(self.index_images, self.index_heatmaps):
            img.set_data(d)

        # update middle
        for i in range(4):
            self.middle_heatmaps[i] = middle_norms[i]
        for img, d in zip(self.middle_images, self.middle_heatmaps):
            img.set_data(d)

        # update ring
        for i in range(4):
            self.ring_heatmaps[i] = ring_norms[i]
        for img, d in zip(self.ring_images, self.ring_heatmaps):
            img.set_data(d)

        # redraw
        self.tactile_fig.canvas.draw_idle()
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
    return taxels, forces

def extract_finger_sensor_data(sensor: SensorFull):
    pos = sensor.sensor_pos
    taxel_msgs = sensor.taxels
    # convert to numpy array, shape = (4, 4, 3)
    taxels = np.array([[taxel_msg.x, taxel_msg.y, taxel_msg.z] for taxel_msg in taxel_msgs]).reshape((4, 4, 3))
    return taxels


############################################################
if __name__ == '__main__':
    
    rclpy.init(args=None)

    node = TactileListener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()