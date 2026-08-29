# Depth-synchronized palm-facing observations for one RGB frame.
# Kept separate from HandKeypoints so existing external subscribers retain the
# exact original Hand/HandKeypoints type hash.

std_msgs/Header header

string estimator_name
string estimator_version
string estimator_spec_sha256
string calibration_version
string handedness_mapping_version

string[] supported_facings
string rejection_category

HandFacing[] hands
