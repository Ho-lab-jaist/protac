# protac
The repository includes ROS packages and nodes for ProTac sensing and perception (```protac_perception```), as well as control of the ProTac-integrated UR robot and newly constructed robot arm (```protac_control```). The ProTac design files are available in the ```protac_design``` directory.

*Note: The repository is under-reviewed by Transaction on Robotics.*

## ProTac perception
For contact sensing, run the following ROS node
```
rosrun protac_perception contact_sensing_node.py
```
For distance / proximity, run
```
rosrun protac_perception distance_sensing_node.py
```
