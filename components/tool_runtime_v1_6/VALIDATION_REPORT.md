# Validation report — v1.4.0-rc1

검증일: 2026-08-18  
기준 환경: Ubuntu 24.04, ROS2 Jazzy, Python 3.12, RTX 3060 12 GB

## 통과 항목

| 항목 | 결과 |
|---|---|
| 비-ROS synthetic planar pose/RLE contract | PASS |
| synthetic `16UC1 compressedDepth` decode/registration/z-buffer | PASS |
| ROS2 clean `colcon build` | PASS, 2 packages |
| ROS2 unit tests | PASS, 5 tests / 0 failures |
| 대상 Python source `ament_flake8` | PASS |
| bundle-relative node startup | PASS |
| checkpoint load and planar-pose algorithm initialization | PASS |
| reference MCAP CAM4 RGB/depth message scan | PASS, 447/447 frames |
| reference MCAP native depth registration | PASS |

Reference MCAP registration 결과:

```text
depth format: 16UC1; compressedDepth png
depth shape: 720 x 1280
nearest RGB-depth delta median: 0.062988 ms
nearest RGB-depth delta maximum: 0.063233 ms
source valid pixels: 729251
projected points: 729064
RGB-aligned valid pixels: 676054
RGB-aligned valid ratio: 0.7335655382
warm registration median: 약 34.02 ms (3회 실행 중 warm 2회)
```

## 검증 경계

- 실행 환경의 네트워크/DDS socket 권한 제한 때문에 실제 `ros2 bag play` publisher와 node
  subscriber 간 live DDS 수신은 완료하지 못했다. 통합 PC에서 acceptance가 필요하다.
- 별도의 direct serialized-message end-to-end 시험에서는 reference bag 첫 프레임의 RF-DETR
  detection이 confidence 0.5 기준 0개였다. 따라서 해당 프레임에서는 pose array가 비어 있었다.
- 이는 native-depth 입력 파이프라인 실패가 아니라 현재 RF-DETR 모델의 일반화 문제로 관리한다.
- `depth_scale_m_per_unit=0.001`과 support plane은 provisional이므로 metric production acceptance가
  아니다.
