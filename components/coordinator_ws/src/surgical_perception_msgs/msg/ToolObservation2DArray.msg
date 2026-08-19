# Lossless per-instance 2D geometry for exactly one source image.
std_msgs/Header header
uint64 sequence
string schema_version
string observation_id
string view
uint32 image_width
uint32 image_height
string model_version
string ontology_version
ToolObservation2D[] instances
