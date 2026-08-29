from glob import glob
import os

from setuptools import find_packages, setup


package_name = "pnu_surgical_perception"


setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "config", "roi_profiles"),
            glob("config/roi_profiles/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="PNU CV Lab",
    maintainer_email="noreply@example.com",
    description="ROS 2 adapters for PNU surgical-tool perception.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "native_depth_tool_pose = pnu_surgical_perception.native_depth_pose_node:main",
            "debug_overlay_compositor = pnu_surgical_perception.debug_overlay_compositor:main",
            "perception_ingress = pnu_surgical_perception.perception_ingress:main",
            "final_overlay_compositor = pnu_surgical_perception.final_overlay_compositor:main",
            "cam4_palm_pose_transform = pnu_surgical_perception.cam4_palm_pose_transform:main",
            "multiview_hand_fusion = pnu_surgical_perception.multiview_hand_fusion:main",
            "operator_quad_compositor = pnu_surgical_perception.operator_quad_compositor:main",
        ],
    },
)
