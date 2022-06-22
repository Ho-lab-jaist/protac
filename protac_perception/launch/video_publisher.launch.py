import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    package_name = "protac_perception"
    config = os.path.join(
    get_package_share_directory(package_name),
    'config',
    'camera_params.config.yaml'
    )

    img1_publisher = Node(
        package="protac_perception",
        namespace="cam1",
        executable="img_publisher",
        name='img_publisher_node',
        parameters = [config],
    )

    img2_publisher = Node(
        package="protac_perception",
        namespace="cam2",
        executable="img_publisher",
        name='img_publisher_node',
        parameters = [config],
    )

    img3_publisher = Node(
        package="protac_perception",
        namespace="cam3",
        executable="img_publisher",
        name='img_publisher_node',
        parameters = [config],
    )

    img4_publisher = Node(
        package="protac_perception",
        namespace="cam4",
        executable="img_publisher",
        name='img_publisher_node',
        parameters = [config],
    )

    return LaunchDescription([
        # img1_publisher,
        # img2_publisher,
        # img3_publisher,
        img4_publisher,
    ])