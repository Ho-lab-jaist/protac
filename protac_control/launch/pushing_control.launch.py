import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, LogInfo, EmitEvent
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.events import Shutdown

from launch_ros.actions import Node



def generate_launch_description():
    pushing_control = Node(
        package="protac_control",
        executable="pushing_control_node",
        name='pushing_control_node',
    )

    velocity_controller = ExecuteProcess(
        cmd=["ros2", "control", "switch_controllers", "--start", "joint_state_broadcaster",
             "--stop", "joint_trajectory_controller", "--start", "velocity_controller"],
        output="screen"
    )
        
    position_controller = ExecuteProcess(
        cmd=["ros2", "control", "switch_controllers", "--start", "joint_state_broadcaster",
             "--stop", "joint_trajectory_controller", "--start", "position_controller"],
        output="screen"
    )

    joint_trajectory_controller = ExecuteProcess(
        cmd=["ros2", "control", "switch_controllers", "--start", "joint_state_broadcaster",
             "--start", "joint_trajectory_controller", "--stop", "velocity_controller"],
        output="screen"
    )

    return LaunchDescription([

        pushing_control,

        # RegisterEventHandler(
        #     event_handler=OnProcessStart(
        #         target_action=pushing_control,
        #         on_start=[velocity_controller],
        #     )
        # ),

        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=pushing_control,
        #         on_exit=[joint_trajectory_controller]
        #     )
        # ),
    ])

