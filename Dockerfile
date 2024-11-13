ARG CUDA_VER=11.4.3
FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu20.04
#FROM ubuntu:20.04

# Set the timezone for TZ
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install utils
RUN apt-get update
RUN apt-get install -y gnupg2 build-essential cmake git tmux nano curl lsb-release
RUN apt-get install -y v4l-utils

## Install ROS ##########################################
# Set up keys for ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update
# Install ros
RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-noetic-desktop-full
#Setup environment 
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
##Install dependencies
RUN apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN apt-get install -y ros-noetic-ros-controllers ros-noetic-ros-control ros-noetic-rqt-controller-manager
###########################################################

## Install python packages
RUN apt update
RUN apt install -y python3-pip
RUN pip3 install vcstool
COPY requirements.txt /home
WORKDIR /home
RUN pip3 install -r requirements.txt

## Install OpengGl
RUN apt install -y libglfw3-dev libglm-dev
RUN apt install -y mesa-utils

## Configure ROS environment and add UR5 packages
WORKDIR /home
RUN mkdir -p /home/catkin_ws/src
WORKDIR /home/catkin_ws/
RUN git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver
RUN git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git src/universal_robot
RUN apt update -qq
RUN rosdep init
RUN rosdep update
RUN /bin/bash -c "source /opt/ros/noetic/setup.sh && rosdep install --from-paths src --ignore-src -y"
RUN /bin/bash -c "source /opt/ros/noetic/setup.sh && catkin_make"
# Load the modified configuration files
COPY ./ur5e_config/ur_control.launch /home/catkin_ws/src/Universal_Robots_ROS_Driver/ur_robot_driver/launch/ur_control.launch
COPY ./ur5e_config/ur5e_bringup.launch /home/catkin_ws/src/Universal_Robots_ROS_Driver/ur_robot_driver/launch/ur5e_bringup.launch
COPY ./ur5e_config/holab_ur5e_calibration.yaml /home/catkin_ws/src/universal_robot/ur_description/config/ur5e/default_kinematics.yaml

## Install UR5 Ros Packages
RUN apt-get install -y ros-noetic-universal-robots
# Install Library to process skin data
WORKDIR /tmp/
COPY resources/robotskin-0.1-amd64.deb .
COPY resources/pyrobotskin-0.1.0-cp38-cp38-linux_x86_64.whl .
RUN dpkg -i robotskin-0.1-amd64.deb
RUN pip3 install pyrobotskin-0.1.0-cp38-cp38-linux_x86_64.whl
COPY resources/taclink.json .

# Install KDL
WORKDIR /home/catkin_ws/src
RUN git clone https://github.com/amir-yazdani/hrl-kdl.git
WORKDIR /home/catkin_ws/src/hrl-kdl/
RUN git checkout noetic-devel
WORKDIR /home/catkin_ws/src/hrl-kdl/pykdl_utils
RUN python3 setup.py build
RUN python3 setup.py install
WORKDIR /home/catkin_ws/src/hrl-kdl/hrl_geom/
RUN python3 setup.py build
RUN python3 setup.py install
RUN apt install -y ros-noetic-urdf-parser-plugin
RUN apt install -y ros-noetic-urdfdom-py
#RUN apt install -y ros-noetic-kdl-parser
#RUN apt install -y ros-noetic-kdl-parser-py
WORKDIR /home/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.sh && catkin_make"

# Clone KDL parser
WORKDIR /home/catkin_ws/src
RUN git clone https://github.com/ros/kdl_parser.git
# Copy the fixed kdl_kinematic file
COPY ./ur5e_config/kdl_kinematics.py ./hrl-kdl/pykdl_utils/src/pykdl_utils/
WORKDIR /home/catkin_ws/
RUN /bin/bash -c "source /opt/ros/noetic/setup.sh && catkin_make"

# Set the workdir
WORKDIR /home/catkin_ws

# Clean the apt cache
RUN rm -rf /var/lib/apt/lists/*