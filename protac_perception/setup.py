from setuptools import setup
import glob
import os

package_name = 'protac_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='QuanLuu',
    maintainer_email='quan-luu@jaist.ac.jp',
    description='High-level perception for the ProTac arm',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'img_publisher = protac_perception.img_pub:main',
            'contact_sensing_node = protac_perception.contact_sensing_node:main',
            'obstacle_sensing_node = protac_perception.obstacle_sensing_node:main',
            'distance_sensing_node = protac_perception.distance_sensing_node:main',
        ],
    },
)
