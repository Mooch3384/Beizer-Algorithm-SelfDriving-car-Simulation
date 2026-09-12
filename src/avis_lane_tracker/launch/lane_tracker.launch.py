import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('avis_lane_tracker')

    # Path to config files
    default_params_file = os.path.join(pkg_share, 'config', 'params.yaml')
    default_rviz_file = os.path.join(pkg_share, 'config', 'rviz_config.rviz')

    # Declare Launch Arguments
    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to ROS 2 parameters file'
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_file,
        description='Path to RViz 2 configuration file'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz 2'
    )

    # 1. AvisEngine Simulator Bridge Node
    bridge_node = Node(
        package='avis_lane_tracker',
        executable='avis_sim_bridge',
        name='avis_sim_bridge',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    # 2. Lane Tracker & Local Mapping Node
    tracker_node = Node(
        package='avis_lane_tracker',
        executable='lane_tracker_node',
        name='lane_tracker_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )

    # 3. RViz 2 Visualization Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=None
    )

    return LaunchDescription([
        params_arg,
        rviz_arg,
        use_rviz_arg,
        bridge_node,
        tracker_node,
        rviz_node
    ])
