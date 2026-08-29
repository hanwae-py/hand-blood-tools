# Shared camera selector. VIPLab publishes /synced/<cam>/... ; workers
# subscribe to /perception/ingress/<cam>/... after apply_ingress_cam.
# Source from a run script after config/system.env.
# Any /synced name works: cam_1, cam_2, cam_3, cam_4, flir.
# Also accepts 1, cam1, or /synced/cam_1. CAM1 and CAM2 support RGB-D with an
# RGB-only fallback until registration inputs are usable; FLIR is RGB.

_RESERVED_CAM_WORDS='^(all|help|tool|hand|blood|true|false)$'

is_camera_selector() {
  local raw="${1:-}"
  [[ -z "${raw}" || "${raw}" == -* ]] && return 1
  raw="${raw#/synced/}"
  raw="${raw%%/*}"
  [[ "${raw}" =~ ${_RESERVED_CAM_WORDS} ]] && return 1
  [[ "${raw}" =~ ^[0-9]+$ ]] && return 0
  [[ "${raw}" =~ ^cam[0-9]+$ ]] && return 0
  [[ "${raw}" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]]
}

normalize_cam_name() {
  local raw="$1"
  raw="${raw#/synced/}"
  raw="${raw%%/*}"
  if [[ "${raw}" =~ ^[0-9]+$ ]]; then
    printf 'cam_%s' "${raw}"
  elif [[ "${raw}" =~ ^cam[0-9]+$ ]]; then
    printf 'cam_%s' "${raw#cam}"
  else
    printf '%s' "${raw}"
  fi
}

apply_synced_cam() {
  CAM="$(normalize_cam_name "$1")"
  if [[ ! "${CAM}" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]]; then
    echo "invalid camera id: $1" >&2
    return 1
  fi
  COLOR_TOPIC="/synced/${CAM}/color/image_raw/compressed"
  COLOR_CAMERA_INFO_TOPIC="/synced/${CAM}/color/camera_info"
  DEPTH_TOPIC="/synced/${CAM}/depth/image_rect_raw/compressedDepth"
  DEPTH_CAMERA_INFO_TOPIC="/synced/${CAM}/depth/camera_info"
  EXTRINSICS_TOPIC="/synced/${CAM}/extrinsics/depth_to_color"
}

apply_ingress_cam() {
  apply_synced_cam "$1" || return
  local prefix="/perception/ingress/${CAM}"
  COLOR_TOPIC="${prefix}/color/image_raw/compressed"
  COLOR_CAMERA_INFO_TOPIC="${prefix}/color/camera_info"
  DEPTH_TOPIC="${prefix}/depth/image_rect_raw/compressedDepth"
  DEPTH_CAMERA_INFO_TOPIC="${prefix}/depth/camera_info"
  EXTRINSICS_TOPIC="${prefix}/extrinsics/depth_to_color"
}

list_synced_cameras() {
  local raw
  raw="$(ros2 topic list --no-daemon --spin-time "${SPIN_TIME:-1}" 2>/dev/null || true)"
  printf '%s\n' "${raw}" | sed -n 's|^/synced/\([^/]*\)/.*|\1|p' | sort -u
}

print_synced_cameras() {
  local cams
  cams="$(list_synced_cameras || true)"
  if [[ -n "${cams}" ]]; then
    printf 'published /synced cameras: %s\n' "$(printf '%s' "${cams}" | tr '\n' ' ')"
  else
    printf 'published /synced cameras: (none yet; cam_1 cam_2 cam_3 cam_4 flir are valid)\n'
  fi
}
