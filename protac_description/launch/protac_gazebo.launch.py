import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import xacro
import yaml

def generate_launch_description():
    robot_name = "protac"
    package_name = robot_name + "_description"
    rviz_config = os.path.join(get_package_share_directory(
        package_name), "launch", robot_name + ".rviz")
    robot_description = os.path.join(get_package_share_directory(
        package_name), "urdf", robot_name + "_gazebo.urdf.xacro")
    robot_description_config = xacro.process_file(robot_description)

    # robot_description_config = xacro.parse(open(robot_description))
    # xacro.process_doc(robot_description_config)

    waypoints_file = os.path.join(
        get_package_share_directory(
            package_name), "controllers", "waypoints.yaml"
    )
    with open(waypoints_file) as file:
        wayponits_data = yaml.full_load(file)

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description",
                   "-entity", "protac"],
        output="screen"
    )

    rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output={
                "stdout": "screen",
                "stderr": "log",
            },
    )

    # Launching gazebo.launch.py is comsumed more than 1 minute somehow...
    joint_state_broadcaster = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "start", "--spin-time", "120",
             "joint_state_broadcaster"],
        output="screen"
    )

    joint_trajectory_controller = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "start",
             "joint_trajectory_controller"],
        output="screen"
    )

    velocity_controller = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "configure",
             "velocity_controller"],
        output="screen"
    )

    position_controller = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "configure",
             "position_controller"],
        output="screen"
    )

    send_goal = ExecuteProcess(
        cmd=["ros2", "action", "send_goal", "/joint_trajectory_controller/follow_joint_trajectory", 
             "control_msgs/action/FollowJointTrajectory", "-f",
             "{0}".format(wayponits_data)
            ],
        output="screen"
    )

    return LaunchDescription([
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[joint_state_broadcaster],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster,
                on_exit=[joint_trajectory_controller],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_trajectory_controller,
                on_exit=[velocity_controller],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=velocity_controller,
                on_exit=[position_controller],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=position_controller,
                on_exit=[send_goal],
            )
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory("gazebo_ros"), "launch"), "/gazebo.launch.py"]),
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[
                {"robot_description": robot_description_config.toxml()}],
            output="screen"),

        spawn_entity,

        rviz_node

    ])
