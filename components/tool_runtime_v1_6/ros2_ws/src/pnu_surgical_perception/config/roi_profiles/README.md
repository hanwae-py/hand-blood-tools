# Tool workspace ROI profiles

ROI coordinates are camera-view specific. Keep them separate from the camera transport/depth
calibration YAML so rosbag datasets and live ROS tests can explicitly select the matching view.

Profile names must contain lowercase letters, numbers, underscores, or hyphens and must begin
with the camera name (`cam3_` or `cam4_`). Each profile uses source-image normalized `(x, y)`
coordinates.

Live ROS example:

```bash
TOOL_MODEL_SIZE=xlarge \
TOOL_ROI_PROFILE=cam4_20260814_mayo \
bash scripts/run_tool_v16.sh cam_4
```

The bundled CAM4 reference-rosbag runner selects `cam4_20260814_mayo` by default. Override it
with `TOOL_ROI_PROFILE=none` or another matching profile. The general live runner defaults to
`none`, which leaves ROI filtering disabled.

For a new view, copy an existing profile, change `workspace_roi_profile` and the polygon, and
validate it against that dataset or fixed live-camera installation. Never silently reuse a
profile after the camera or work surface moves.
