# ROS 2 Multi-PC Transmission Test

Date: 2026-08-11
Status: **Basic ROS 2 transmission confirmed working**

## Purpose

Verify ROS 2 communication from Bigyeol's RF-DETR surgical-tool
publisher PC to Nay's temporary subscriber PC before testing the complete
detection-result topics.

```text
Bigyeol PC (publisher, ROS 2 Jazzy, 10.126.34.33)
                         |
                         | ROS 2 DDS over LAN
                         v
Nay PC (subscriber, ROS 2 Jazzy/WSL2, 10.126.34.34)
```

The actual receiver used ROS 2 Jazzy. An earlier plan mentioned a Humble
receiver, but that was not the environment used for this successful basic
test.

## Shared ROS configuration

| Setting | Value |
| --- | --- |
| Publisher IPv4 | `10.126.34.33` |
| Subscriber IPv4 | `10.126.34.34` |
| Subnet | `10.126.34.0/24` |
| `ROS_DOMAIN_ID` | `77` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| Discovery | Reciprocal static peers |

## WSL networking requirement

Nay's Windows file `%USERPROFILE%\.wslconfig` contains:

```ini
[wsl2]
networkingMode=mirrored
```

After changing this file, restart WSL from Windows PowerShell:

```powershell
wsl --shutdown
wsl -d Ubuntu
```

Mirrored mode is required here so WSL can participate directly in LAN
multicast/unicast traffic instead of being isolated behind the default
WSL NAT network.

Reference: <https://learn.microsoft.com/windows/wsl/networking>

## Scoped Windows firewall rules

The Ethernet connection was classified as `Public`. Reciprocal inbound
rules were added for the Domain 77 DDS UDP range, restricted to the other
test PC.

On Nay's PC, run in **Administrator PowerShell**:

```powershell
New-NetFirewallRule `
  -DisplayName "ROS2 DDS Domain 77 from Bigyeol" `
  -Direction Inbound `
  -Action Allow `
  -Protocol UDP `
  -LocalPort 26650-26999 `
  -RemoteAddress 10.126.34.33 `
  -Profile Public
```

On Bigyeol's PC, run in **Administrator PowerShell**:

```powershell
New-NetFirewallRule `
  -DisplayName "ROS2 DDS Domain 77 from Nay" `
  -Direction Inbound `
  -Action Allow `
  -Protocol UDP `
  -LocalPort 26650-26999 `
  -RemoteAddress 10.126.34.34 `
  -Profile Public
```

These rules are intentionally limited to one remote IP and the UDP range
used for this ROS domain rather than allowing arbitrary inbound traffic.

## ROS environment: Nay subscriber

Open Ubuntu WSL and run:

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_STATIC_PEERS='10.126.34.33'
```

`ROS_LOCALHOST_ONLY` is deprecated in Jazzy. The newer discovery variables
above explicitly permit the remote publisher as a static peer.

## ROS environment: Bigyeol publisher

On the publisher PC:

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_STATIC_PEERS='10.126.34.34'
```

Every ROS process involved in the test must be started from a terminal
that has these values. Exporting them after a process starts does not
change that process's DDS configuration.

Reference: <https://docs.ros.org/en/rolling/Tutorials/Advanced/Improved-Dynamic-Discovery.html>

## Basic transmission test

Stop any stale ROS daemon first on both PCs:

```bash
ros2 daemon stop
```

If it hangs, press `Ctrl+C`. Use `--no-daemon` for graph inspection.

On Bigyeol's PC, publish example messages:

```bash
ros2 run demo_nodes_cpp talker
```

On Nay's PC, receive them:

```bash
ros2 run demo_nodes_py listener
```

Successful output observed on Nay's PC:

```text
[listener]: I heard: [Hello World: 58]
[listener]: I heard: [Hello World: 59]
[listener]: I heard: [Hello World: 60]
[listener]: I heard: [Hello World: 61]
[listener]: I heard: [Hello World: 62]
[listener]: I heard: [Hello World: 63]
[listener]: I heard: [Hello World: 64]
```

This confirms that ROS 2 discovery and message delivery work between the
two PCs. Using a C++ talker and Python listener is valid: ROS messages are
language-independent.

## RF-DETR published-topic contract

| Topic | Type | QoS | Content |
| --- | --- | --- | --- |
| `/perception/cam_4/tool/health` | `std_msgs/msg/String` | Publisher contract | Health JSON |
| `/perception/cam_4/tool/semantics` | `std_msgs/msg/String` | Reliable, Volatile, depth 10 | Class, confidence, bbox, COCO RLE mask and mask centroid |
| `/perception/cam_4/tool/overlay/compressed` | `sensor_msgs/msg/CompressedImage` | Best Effort, Volatile, depth 5 | JPEG overlay containing mask, bbox, class, confidence and centroid |
| `/perception/cam_4/tool/diagnostics` | `std_msgs/msg/String` | Publisher contract | Diagnostics JSON |

All types are standard ROS messages, so the subscriber does not need a
custom interface package.

## Start the RF-DETR publisher

After the basic talker/listener test succeeds, stop the talker with
`Ctrl+C`. Bigyeol starts the bridge from the same configured environment:

```bash
cd /home/user/Projects/SurgicalTool

bash ROS/scripts/run_cam4_rfdetr_bridge.sh \
  -p input_mode:=compressed
```

In another configured publisher terminal, publish the test image:

```bash
source /opt/ros/jazzy/setup.bash
source /home/user/Projects/SurgicalTool/ROS/ros2_ws/install/setup.bash

ros2 run pnu_surgical_perception cam4_image_publisher --ros-args \
  -p image_path:=/home/user/Projects/SurgicalTool/cam4_seg8_local_20260810/test_data/cam4_validation_sample.jpg \
  -p publish_hz:=2.0
```

The publisher must also retain the Domain 77, Fast DDS and reciprocal
static-peer environment when these processes start.

## Verify RF-DETR topics from Nay's PC

Discover the topics:

```bash
ros2 topic list --no-daemon | grep perception
```

Receive health:

```bash
ros2 topic echo /perception/cam_4/tool/health --once --no-daemon
```

Receive one structured detection result:

```bash
ros2 topic echo /perception/cam_4/tool/semantics --once --no-daemon
```

Inspect publisher type and QoS:

```bash
ros2 topic info --verbose /perception/cam_4/tool/semantics --no-daemon
ros2 topic info --verbose /perception/cam_4/tool/overlay/compressed --no-daemon
```

View the overlay:

```bash
ros2 run rqt_image_view rqt_image_view
```

Select:

```text
/perception/cam_4/tool/overlay/compressed
```

If the viewer is not installed:

```bash
sudo apt update
sudo apt install ros-jazzy-rqt-image-view
```

## RF-DETR success criteria

The application-level test succeeds when all three are confirmed on
Nay's PC:

1. Health JSON is received.
2. Semantics JSON contains detection `instances`, mask/RLE and centroid.
3. The compressed detection-overlay image is displayed.

The basic talker/listener result proves the network path. It does not by
itself prove that the RF-DETR bridge started with the same ROS environment
or that all four application topics are publishing.

## Troubleshooting

### Ping works but no ROS topics appear

- Confirm mirrored WSL networking on both Windows PCs.
- Confirm reciprocal scoped firewall rules.
- Confirm both sides use Domain 77 and `rmw_fastrtps_cpp`.
- Confirm each process was started after exporting the environment.
- Confirm the static peer is the other PC's LAN IP.
- If a publisher runs in Docker, confirm its DDS network is not isolated.

### Topics appear locally on the publisher but not remotely

Run the basic talker/listener test. If that succeeds, the network is
working and the RF-DETR bridge likely started from a terminal with a
different ROS environment. Restart the bridge and image publisher after
exporting the shared configuration.

### Plain `ros2 topic list` is empty or stale

The ROS daemon may have started with a different Domain ID/RMW or may be
occupying the first DDS participant port. Stop it and query directly:

```bash
ros2 daemon stop
ros2 topic list --no-daemon
```

### Remove the temporary firewall rules

On Nay's PC, Administrator PowerShell:

```powershell
Remove-NetFirewallRule -DisplayName "ROS2 DDS Domain 77 from Bigyeol"
```

On Bigyeol's PC, Administrator PowerShell:

```powershell
Remove-NetFirewallRule -DisplayName "ROS2 DDS Domain 77 from Nay"
```
