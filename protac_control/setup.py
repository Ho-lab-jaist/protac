from setuptools import setup
import os
from glob import glob

package_name = 'protac_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name,]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('resource/*.csv')),
        (os.path.join('share', package_name), glob('resource/*.vtk')),
        (os.path.join('share', package_name), glob('resource/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='QuanLuu',
    maintainer_email='quan-luu@jaist.ac.jp',
    description='High-level control for the ProTac arm',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_space_control_node = protac_control.joint_space_control_node:main',
            'position_control_node = protac_control.position_control_node:main',
            'velocity_control_node = protac_control.velocity_control_node:main',
            'online_position_control_node = protac_control.online_position_control_node:main',
            'human_aware_speed_control_node = protac_control.human_aware_speed_control_node:main',
            'distance_reactive_control_node = protac_control.distance_reactive_control_node:main',
            'reflex_control_node = protac_control.reflex_control_node:main',
            'distance_aware_speed_control_node = protac_control.distance_aware_speed_control_node:main',
        ],
    },
)
