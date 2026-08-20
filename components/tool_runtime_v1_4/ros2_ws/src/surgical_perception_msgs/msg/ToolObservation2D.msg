# Mask encoding. Zero is deliberately unusable by default.
uint8 MASK_ENCODING_INVALID=0
uint8 MASK_ENCODING_COCO_RLE_COMPRESSED=1

# This ID is unique only inside the parent observation/image frame.
uint32 frame_local_instance_id

uint16 canonical_class_id
uint16 model_class_index
string class_name
float32 class_confidence
float32 segmentation_confidence

# Pixel coordinates use u right, v down, and bbox order x_min,y_min,x_max,y_max.
float32[4] bbox_xyxy_px
float32[4] mask_bbox_xyxy_px
uint32 mask_area_px

# 2D projection of the selected 3D tool observation point P_obs.
# This is deliberately not the arithmetic mask centroid.
float32[2] observation_point_uv_px
bool observation_point_valid
bool observation_point_inside_mask
bool observation_point_depth_valid
float32 observation_point_depth_m
string observation_point_selection_mode
float32 observation_point_boundary_clearance_px

# COCO compressed RLE uses size mask_height x mask_width and column-major order.
uint8 mask_encoding
uint32 mask_height
uint32 mask_width
string mask_counts
