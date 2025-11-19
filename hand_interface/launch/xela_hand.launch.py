from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        # tactile listener node
        Node(
            package='hand_interface',
            executable='tactile_listener.py',
            name='tactile_listener',
            output='screen'
        ),

        # hand control test node
        # Node(
        #     package='hand_interface',
        #     executable='hand_control_test.py',
        #     name='hand_controller',
        #     output='screen'
        # ),
    ])
