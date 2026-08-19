import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'surgical_task_coordinator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Han Nwae Nyein',
    maintainer_email='hannwaenyein2303@gmail.com',
    description='Take-turn task coordinator for the surgical robot (tool / hand / blood).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_coordinator = surgical_task_coordinator.task_coordinator:main',
            'perception_mode_coordinator = surgical_task_coordinator.perception_mode_coordinator:main',
            'stub_detector = surgical_task_coordinator.stub_detector:main',
            'v14_tool_lifecycle_gate = surgical_task_coordinator.v14_tool_lifecycle_gate:main',
            'fake_robot_node = surgical_task_coordinator.fake_robot_node:main',
        ],
    },
)
