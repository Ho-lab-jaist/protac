import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    package_name = "protac_perception"
    # config = os.path.join(
    # get_package_share_directory(package_name),
    # 'config',
    # 'depth_sensing.config.yaml'
    # )


    img1_show = Node(
        package="protac_perception",
        namespace="cam1",
        executable="image_display_node",
        name='image_display_node',
        remappings=[
        ('/video_frames', '/cam1/video_frames'),
        ]
    )

    img2_show = Node(
        package="protac_perception",
        namespace="cam2",
        executable="image_display_node",
        name='image_display_node',
        remappings=[
        ('/video_frames', '/cam2/video_frames'),
        ]
    )

    img3_show = Node(
        package="protac_perception",
        namespace="cam3",
        executable="image_display_node",
        name='image_display_node',
        remappings=[
        ('/video_frames', '/cam3/video_frames'),
        ]
    )

    img4_show = Node(
        package="protac_perception",
        namespace="cam4",
        executable="image_display_node",
        name='image_display_node',
        remappings=[
        ('/video_frames', '/cam4/video_frames'),
        ]
    )


    return LaunchDescription([
        # img1_show,
        # img2_show,
        img3_show,
        # img4_show,
    ])