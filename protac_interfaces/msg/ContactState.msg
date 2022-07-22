# This is a message to hold data for a contact state from a large-scale tactile Sensor
# which covers multi-contact scenarios/conditions

std_msgs/Header header
float64[] contact_depth
float64[] contact_force
geometry_msgs/Vector3[] contact_location
geometry_msgs/Vector3[] contact_direction
int64 num_of_contacts