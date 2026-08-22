#!/usr/bin/env bash
# Show JPEG overlays for any /synced camera.
#   bash scripts/view_overlay.sh cam_3 tool
#   bash scripts/view_overlay.sh flir tool
#   bash scripts/view_overlay.sh all tool
#   bash scripts/view_overlay.sh cam_4 tool hand
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
source "${ROOT}/scripts/select_cam.sh"

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

CAMS=()
if [[ "${1:-}" == "all" ]]; then
  shift
  mapfile -t CAMS < <(list_synced_cameras)
  if [[ ${#CAMS[@]} -eq 0 ]]; then
    echo "no published /synced cameras" >&2
    exit 1
  fi
elif is_camera_selector "${1:-}"; then
  apply_synced_cam "$1"
  CAMS=("${CAM}")
  shift
else
  echo "usage: bash scripts/view_overlay.sh <cam|all> [tool] [hand] [blood]" >&2
  print_synced_cameras >&2
  exit 1
fi

TASKS=("${@:-tool}")
if [[ ${#TASKS[@]} -eq 1 && "${TASKS[0]}" == "all" ]]; then
  TASKS=(tool hand blood)
fi

python3 - "${CAMS[@]}" -- "${TASKS[@]}" <<'PY'
import sys
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

sep = sys.argv.index("--")
cams = sys.argv[1:sep]
tasks = sys.argv[sep + 1:]
topics = {
    f"{cam}/{task}": f"/perception/{cam}/{task}/overlay/compressed"
    for cam in cams
    for task in tasks
}


class OverlayViewer(Node):
    def __init__(self):
        super().__init__("overlay_cv")
        self.frames = {}
        for name, topic in topics.items():
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, n=name: self._on_image(n, msg),
                qos_profile_sensor_data,
            )

    def _on_image(self, name, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            self.frames[name] = img


rclpy.init()
node = OverlayViewer()
try:
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)
        for name, img in node.frames.items():
            cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    cv2.destroyAllWindows()
PY
