# ROS2 Jazzy hand-detection node ("section 2" of the SurgBlood-style ROS2
# flow: camera topics -> AI detection -> results topic). Jazzy officially
# targets Ubuntu 24.04, hence a separate container from the rest of this
# repo's pose_env_gpu (Ubuntu 22.04) tooling.
#
# Build from the hand_keypoints/ repo root (needs both scripts/ and
# ros2_ws/ in the build context):
#   cd hand_keypoints
#   docker build -f ros2_ws/Dockerfile -t hand_keypoint_ros .
#
# Run (GPU):
#   docker run --rm -it --gpus all --network host hand_keypoint_ros
# Run (CPU only, e.g. no GPU on this machine):
#   docker run --rm -it --network host -e CUDA_VISIBLE_DEVICES= hand_keypoint_ros

FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ros-jazzy-cv-bridge \
        ros-jazzy-message-filters \
        ros-jazzy-vision-opencv \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# mediapipe==0.10.18 requires numpy<2, which is older than the numpy>=2
# that hand_keypoints/requirements.txt's opencv-python==4.13.0.92 needs --
# so opencv-python and numpy are left unpinned here and resolved by pip
# to whatever satisfies mediapipe (this differs from requirements.txt,
# which targets a plain venv without mediapipe's numpy<2 constraint).
# --break-system-packages: Ubuntu 24.04's system Python is PEP 668
# "externally managed"; ROS itself already lives in system site-packages,
# so a venv would fight with rclpy's own install instead of helping.
RUN pip install --no-cache-dir --break-system-packages \
        opencv-python \
        mediapipe==0.10.18 \
        torch==2.11.0 \
        transformers==5.12.1 \
        pillow==12.1.1 \
        h5py==3.16.0

# scripts/hand_keypoints_core.py is the single source of truth for the
# detection math (shared with the offline CLI pipeline) -- copied into the
# ROS package at image-build time so the package stays self-contained/
# installable without needing the rest of this git repo mounted at runtime.
# hand_keypoint_interfaces (custom .msg types) has no such dependency --
# it's built straight from ros2_ws/src, nothing to copy in.
COPY scripts/hand_keypoints_core.py /tmp/hand_keypoints_core.py
COPY ros2_ws/src /ros_ws/src
RUN cp /tmp/hand_keypoints_core.py /ros_ws/src/hand_keypoint_ros/hand_keypoint_ros/core.py

WORKDIR /ros_ws
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
 && echo "source /ros_ws/install/setup.bash" >> /root/.bashrc

SHELL ["/bin/bash", "-c"]
CMD ["bash"]
