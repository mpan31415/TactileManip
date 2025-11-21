#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from xela_server_ros2.msg import SensStream, SensorFull

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


############################################################
class TactileListenerNode(Node):

    def __init__(self):
        super().__init__('tactile_listener_node')

        # subscribe to palm & finger sensors
        depth = 5
        self.palm_sub = self.create_subscription(SensStream, 'xPalmTopic', self.palm_cb, depth)
        self.fingers_sub = self.create_subscription(SensStream, 'xFingersTopic', self.fingers_cb, depth)

        # tactile data buffer
        self.palm_forces = np.zeros((3, 4, 6, 3))
        self.palm_taxels = np.zeros((3, 4, 6, 3))
        self.finger_taxels = np.zeros((15, 4, 4, 3))

        # read taxel mean
        self.palm_taxel_mean, self.finger_taxel_mean = get_taxel_mean()    # shapes: (), (15, 4, 4, 3)
        self.palm_taxel_stddev = 500.0       
        self.finger_taxel_stddev = 250.0       # TODO: calibrate/estimate?

        # palm data plotter
        plot_update_freq = 20.0    # Hz
        self.timer = self.create_timer(1 / plot_update_freq, self.update_tactile_plot)

        # initialize plots
        self.init_tactile_plot()
        
        
    def palm_cb(self, msg: SensStream):
        # Palm sensor pos ordering:
        # 1 - top right
        # 2 - top left
        # 3 - bottom left
        sensors = msg.sensors
        for i in range(3):
            taxels, forces = extract_palm_sensor_data(sensors[i])
            # normalize raw taxel data
            normalized_taxels = (taxels - self.palm_taxel_mean[i, :, :, :]) / self.palm_taxel_stddev

            # apply rotation to align
            if i+1 in [2, 3]:
                normalized_taxels = np.rot90(normalized_taxels, 2, axes=(0, 1))
                forces = np.rot90(forces, 2, axes=(0, 1))
            
            # store
            self.palm_forces[i, :, :, :] = forces
            self.palm_taxels[i, :, :, :] = normalized_taxels


    def fingers_cb(self, msg: SensStream):
        # Fingers sensor pos ordering:
        # [1, 2, 3] - thumb (tip to base)
        # [4, 5, 6, 7] - index (tip to base)
        # [8, 9, 10, 11] - middle (tip to base)
        # [12, 13, 14, 15] - ring (tip to base)
        sensors = msg.sensors
        for i in range(15):
            taxels = extract_finger_sensor_data(sensors[i])
            # normalize raw taxel data
            normalized_taxels = (taxels - self.finger_taxel_mean[i, :, :, :]) / self.finger_taxel_stddev

            # apply rotation to align
            if i+1 in [1, 4, 8, 12]:       # fingertips: tranpose
                normalized_taxels = np.transpose(normalized_taxels, (1, 0, 2))
            elif i+1 in [2, 5, 9, 13]:     # 2nd from tip: rotate clockwise by 90 deg
                normalized_taxels = np.rot90(normalized_taxels, -1, axes=(0, 1))
            elif i+1 in [3, 6, 10, 14]:    # 3rd from tip: rotate anticlockwise by 90 deg
                normalized_taxels = np.rot90(normalized_taxels, 1, axes=(0, 1))
            elif i+1 in [7, 11, 15]:       # base (excluding thumb): rotate clockwise by 90 deg
                normalized_taxels = np.rot90(normalized_taxels, -1, axes=(0, 1))

            # store
            self.finger_taxels[i, :, :, :] = normalized_taxels
            
    
    def init_tactile_plot(self):
        # Turn off constrained_layout / tight_layout to avoid fighting with manual layout
        plt.ion()
        self.tactile_fig = plt.figure(figsize=(6, 8), constrained_layout=False)

        # -----------------------------
        # GridSpec layout (hand shape)
        # -----------------------------
        # Use an explicit grid and set equal width/height ratios so cells are uniform.
        nrows, ncols = 28, 21
        gs = GridSpec(nrows, ncols, figure=self.tactile_fig, wspace=0.3, hspace=0.3)

        # ---- Thumb: 3 heatmaps ----
        thumb_ax1 = self.tactile_fig.add_subplot(gs[12:16, 17:21])
        thumb_ax2 = self.tactile_fig.add_subplot(gs[17:21, 16:20])
        thumb_ax3 = self.tactile_fig.add_subplot(gs[22:26, 15:19])
        self.thumb_axes = [thumb_ax1, thumb_ax2, thumb_ax3]

        # store finger axes in a dict
        finger_positions = ["index", "middle", "ring"]
        self.finger_axes = {k: [] for k in finger_positions}

        # ---- Index: 4 heatmaps ----
        index_ax1 = self.tactile_fig.add_subplot(gs[0:4, 10:14])
        index_ax2 = self.tactile_fig.add_subplot(gs[5:9, 10:14])
        index_ax3 = self.tactile_fig.add_subplot(gs[10:14, 10:14])
        index_ax4 = self.tactile_fig.add_subplot(gs[15:19, 10:14])
        self.finger_axes["index"] = [index_ax1, index_ax2, index_ax3, index_ax4]

        # ---- Middle: 4 heatmaps ----
        middle_ax1 = self.tactile_fig.add_subplot(gs[0:4, 5:9])
        middle_ax2 = self.tactile_fig.add_subplot(gs[5:9, 5:9])
        middle_ax3 = self.tactile_fig.add_subplot(gs[10:14, 5:9])
        middle_ax4 = self.tactile_fig.add_subplot(gs[15:19, 5:9])
        self.finger_axes["middle"] = [middle_ax1, middle_ax2, middle_ax3, middle_ax4]

        # ---- Ring: 4 heatmaps ----
        ring_ax1 = self.tactile_fig.add_subplot(gs[0:4, 0:4])
        ring_ax2 = self.tactile_fig.add_subplot(gs[5:9, 0:4])
        ring_ax3 = self.tactile_fig.add_subplot(gs[10:14, 0:4])
        ring_ax4 = self.tactile_fig.add_subplot(gs[15:19, 0:4])
        self.finger_axes["ring"] = [ring_ax1, ring_ax2, ring_ax3, ring_ax4]

        # ---- Palm: 3 heatmaps ----
        palm_ax1 = self.tactile_fig.add_subplot(gs[20:24, 7:14])
        palm_ax2 = self.tactile_fig.add_subplot(gs[20:24, 0:7])
        palm_ax3 = self.tactile_fig.add_subplot(gs[24:28, 0:7])
        self.palm_axes = [palm_ax1, palm_ax2, palm_ax3]

        # -----------------------------
        # Initialize heatmaps (data shapes)
        # -----------------------------
        self.palm_heatmaps = [np.zeros((4, 6)) for _ in range(3)]
        self.thumb_heatmaps = [np.zeros((4, 4)) for _ in range(3)]
        self.index_heatmaps = [np.zeros((4, 4)) for _ in range(4)]
        self.middle_heatmaps = [np.zeros((4, 4)) for _ in range(4)]
        self.ring_heatmaps = [np.zeros((4, 4)) for _ in range(4)]

        # Helper to make each axis "fill" its GridSpec cell and not try to force aspect ratio
        def init_image(ax, data, vmin=0.0, vmax=5.0):
            img = ax.imshow(data, vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            # important: make axis fill the box allocated by GridSpec
            ax.set_aspect('auto')
            ax.set_adjustable('box')       # ensure the box is adjusted (not the data limits)
            ax.set_anchor('C')            # center anchor to avoid odd offsets
            return img

        # Create images
        self.palm_images = [init_image(ax, d) for ax, d in zip(self.palm_axes, self.palm_heatmaps)]
        self.thumb_images = [init_image(ax, d) for ax, d in zip(self.thumb_axes, self.thumb_heatmaps)]
        self.index_images = [init_image(ax, d) for ax, d in zip(self.finger_axes["index"], self.index_heatmaps)]
        self.middle_images = [init_image(ax, d) for ax, d in zip(self.finger_axes["middle"], self.middle_heatmaps)]
        self.ring_images = [init_image(ax, d) for ax, d in zip(self.finger_axes["ring"], self.ring_heatmaps)]

        # final adjust: small margin so outer cells are not cut off
        self.tactile_fig.subplots_adjust(left=0.03, right=0.95, top=0.98, bottom=0.02)

        plt.show()


    def update_tactile_plot(self):
        # data to plot
        palm_norms = np.linalg.norm(self.palm_taxels, axis=3)
        thumb_norms = np.linalg.norm(self.finger_taxels[0:3, :, :, :], axis=3)
        index_norms = np.linalg.norm(self.finger_taxels[3:7, :, :, :], axis=3)
        middle_norms = np.linalg.norm(self.finger_taxels[7:11, :, :, :], axis=3)
        ring_norms = np.linalg.norm(self.finger_taxels[11:15, :, :, :], axis=3)

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


def get_taxel_mean():

    # empty lists
    palm_taxel_mean = []
    finger_taxel_mean = []

    # read data
    for sensor_idx in range(1, 4):
        save_dir = "/home/mpan31415/ros2_ws/src/TactileManip/hand_interface/scripts/calib/taxel_mean/"
        taxel_mean = np.load(save_dir + f"palm_sensor_{sensor_idx}_mean.npy")
        palm_taxel_mean.append(taxel_mean)
    for sensor_idx in range(1, 16):
        save_dir = "/home/mpan31415/ros2_ws/src/TactileManip/hand_interface/scripts/calib/taxel_mean/"
        taxel_mean = np.load(save_dir + f"finger_sensor_{sensor_idx}_mean.npy")
        finger_taxel_mean.append(taxel_mean)

    # convert to numpy arrays
    palm_taxel_mean = np.array(palm_taxel_mean)  # shape = (3, 4, 6, 3)
    finger_taxel_mean = np.array(finger_taxel_mean)  # shape = (15, 4, 4, 3)

    return palm_taxel_mean, finger_taxel_mean


def extract_palm_sensor_data(sensor: SensorFull):
    taxel_msgs = sensor.taxels
    force_msgs = sensor.forces
    # convert to numpy arrays, shape = (4, 6, 3)
    taxels = np.array([[taxel_msg.x, taxel_msg.y, taxel_msg.z] for taxel_msg in taxel_msgs]).reshape((4, 6, 3))
    forces = np.array([[force_msg.x, force_msg.y, force_msg.z] for force_msg in force_msgs]).reshape((4, 6, 3))
    return taxels, forces

def extract_finger_sensor_data(sensor: SensorFull):
    taxel_msgs = sensor.taxels
    # convert to numpy array, shape = (4, 4, 3)
    taxels = np.array([[taxel_msg.x, taxel_msg.y, taxel_msg.z] for taxel_msg in taxel_msgs]).reshape((4, 4, 3))
    return taxels


############################################################
if __name__ == '__main__':
    
    rclpy.init(args=None)

    node = TactileListenerNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()