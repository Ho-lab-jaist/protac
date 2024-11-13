# protac
The repository includes ROS packages and nodes for ProTac sensing and perception (```protac_perception```), as well as control of the ProTac-integrated UR robot (```ur_protac```) and newly constructed robot arm (```protac_control```). The ProTac design files are available in the ```protac_design``` directory.

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

## ProTac control
The optimization-based motion control with combined obstacle awareness and contact accomoation can be tested from
```
rosrun ur_protac test_protac_qp_control_proximity.py
```

The  human-robot interaction with ProTac flickering sensing can be tested from
```
rosrun protac_control
```

ProTac-driven reactive control and speed regulation can be run from
```
rosrun protac_control
```
```
rosrun protac_control
```
