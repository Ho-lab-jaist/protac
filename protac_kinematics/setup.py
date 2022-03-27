from setuptools import setup
import os
from glob import glob

package_name = 'protac_kinematics'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='QuanLuu',
    maintainer_email='quan-luu@jaist.ac.jp',
    description='protac kinematics helper',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kinematics = protac_kinematics.kinematics:main'
        ],
    },
)
