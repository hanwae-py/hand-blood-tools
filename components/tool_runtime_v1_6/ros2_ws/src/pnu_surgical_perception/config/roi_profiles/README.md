# Tool workspace ROI profiles

ROI coordinates are camera-view specific. Keep them separate from the camera transport/depth
calibration YAML so rosbag datasets and live ROS tests can explicitly select the matching view.

Profile names must contain lowercase letters, numbers, underscores, or hyphens and must begin
with the camera name (`cam3_`, `cam4_`, or `head_`). Each profile uses source-image normalized `(x, y)`
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

The 2026-08-25 RGB recordings use `cam3_20260825_arpa_sharing_tray` and
`cam4_20260825_arpa_sharing_mayo`. These polygons were checked on the supplied RGB videos but
remain provisional because they do not have annotated mask ground truth.

For a new view, copy an existing profile, change `workspace_roi_profile` and the polygon, and
validate it against that dataset or fixed live-camera installation. Never silently reuse a
profile after the camera or work surface moves.

An operator can draw a profile directly on a representative video frame:

```bash
python3 algorithm/validation/select_video_roi.py \
  --video /path/to/cam4.mp4 --frame-time-sec 30 \
  --profile cam4_room1_mayo --workspace-zone mayo \
  --output-yaml ros2_ws/src/pnu_surgical_perception/config/roi_profiles/cam4_room1_mayo.yaml
```

Left click adds a vertex, right click or Backspace removes the last vertex, `R` resets, and
Enter or `S` saves the normalized polygon and a preview image.
