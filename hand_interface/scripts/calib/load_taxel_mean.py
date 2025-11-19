import numpy as np




if __name__ == "__main__":
    
    # get argument from command line for SENSOR_IDX
    import sys
    if len(sys.argv) > 1:
        SENSOR_IDX = int(sys.argv[1])
    else:
        SENSOR_IDX = 1
    print("Using SENSOR_IDX =", SENSOR_IDX)

    save_dir = "/home/mpan31415/ros2_ws/src/TactileManip/hand_interface/scripts/calib/taxel_mean/"
    taxel_mean = np.load(save_dir + f"finger_sensor_{SENSOR_IDX}_mean.npy")

    print(taxel_mean)
