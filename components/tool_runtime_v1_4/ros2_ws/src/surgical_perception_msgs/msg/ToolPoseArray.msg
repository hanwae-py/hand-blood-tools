# One authoritative per-view surgical-tool pose result for a source image.
std_msgs/Header header
uint64 sequence
string schema_version
string observation_id
string source_view
string model_version
string ontology_version
string calibration_version
string pose_convention_version
ToolPose[] tools
