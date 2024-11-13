std_msgs/Header header
# Feedback: skin state ##################
std_msgs/String state
# Feedback: tactile ##################
std_msgs/Float64 max_depth
geometry_msgs/Point xc # Contact centroid
# Feedback: proximity ##################
std_msgs/Float64 risk_score
geometry_msgs/Point obs_location # obstacle point
std_msgs/Float64 closest_distance
std_msgs/Float64 human_detection_score
std_msgs/Bool object_detected