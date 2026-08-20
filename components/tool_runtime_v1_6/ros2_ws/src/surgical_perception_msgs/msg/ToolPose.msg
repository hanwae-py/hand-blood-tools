# Pose modes. Zero is the safe, unusable default.
uint8 POSE_MODE_INVALID=0
uint8 POSE_MODE_POSITION_3D_ONLY=1
uint8 POSE_MODE_PLANAR_4DOF_WITH_NORMAL_PRIOR=2
uint8 POSE_MODE_FULL_6D=3
uint8 POSE_MODE_AMBIGUOUS=4

# Overall validity. Consumers must check this before using pose.
uint8 VALIDITY_INVALID=0
uint8 VALIDITY_VALID=1
uint8 VALIDITY_DEGRADED=2
uint8 VALIDITY_STALE=3

# Instance and canonical class.
uint32 frame_local_instance_id
uint16 canonical_class_id
uint16 model_class_index
string class_name
float32 class_confidence

# pose.position is the selected tool observation point P_obs in metres.
# pose.orientation is quaternion x,y,z,w.
# dof_observed order: x, y, z, roll, pitch, yaw.
geometry_msgs/Pose pose
uint8 pose_mode
bool position_valid
bool orientation_valid
bool[6] dof_observed

# Definition and 2D evidence of P_obs / the attached output tool-frame origin O_T.
# P_obs is not a mask centroid, CAD centre, centre of mass, grasp point, or TCP.
string observation_point_definition
float32[2] observation_point_uv_px
bool observation_point_inside_mask
bool observation_point_depth_valid
string observation_point_selection_mode
float32 observation_point_boundary_clearance_px

# Orientation convention and ambiguity.
string axis_definition
string symmetry_type
float32 endpoint_sign_confidence

# Pose evidence and quality.
float32 valid_depth_ratio
uint32 pose_point_count
float32 axis_anisotropy
float32 support_plane_inlier_ratio
float32 support_plane_residual_p95_m
float32 pose_confidence
bool pose_confidence_calibrated

uint8 validity
string[] status_flags
string invalid_reason
