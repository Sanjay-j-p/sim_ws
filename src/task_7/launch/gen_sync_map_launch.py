from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()

    # Launch SLAM
    slam_launch_file_path = os.path.join(
        get_package_share_directory('turtlebot4_navigation'),
        'launch',
        'slam.launch.py'
    )
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch_file_path),
            launch_arguments={'namespace': '/robot'}.items(),
        )
    )

    # Launch Rviz2 for viewing the map
    view_robot_launch_file_path = os.path.join(
        get_package_share_directory('turtlebot4_viz'),
        'launch',
        'view_robot.launch.py'
    )
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(view_robot_launch_file_path),
            launch_arguments={'namespace': '/robot'}.items(),
        )
    )

    # Run teleop_twist_keyboard node
    teleop_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        output='screen',
        remappings=[
            ('cmd_vel', '/robot/cmd_vel')
        ]
    )
    ld.add_action(teleop_node)

    return ld
