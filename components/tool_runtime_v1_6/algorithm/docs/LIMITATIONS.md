# 제한 및 책임 경계

1. 이 알고리즘은 native depth-to-color registration을 제공하지만 ROS subscription과 pair queue를
   소유하지 않는다. adapter가 timestamp tolerance와 one-to-one pairing을 보장해야 한다.
2. 현재 transport decoder는 기준 MCAP의 `16UC1 compressedDepth PNG`만 지원한다.
   `32FC1` inverse-depth codec은 지원하지 않는다.
3. support plane을 매 frame 자동 추정하지 않는다. 입력 plane의 정확성은 upstream 책임이다.
4. frame-local instance ID는 다음 frame의 동일 물체를 뜻하지 않는다.
5. P_obs는 CAD origin, TCP, grasp point로 변환되지 않는다.
6. 현재 pose는 constrained planar 4DoF이며 unconstrained full 6D가 아니다.
7. CAM4와 tray의 calibration, plane, frame name은 서로 별개다.
8. 현재 model은 CAM4 데이터에서만 검증되었고 tray readiness는 `NOT VALIDATED`다.
9. Hand Pose와 Blood Detection 및 세 알고리즘의 자원 경합은 범위 밖이다.
10. native 1280x720 registration은 추가 CPU 비용이 있으므로 목표 장비에서 strict latency를
    다시 측정해야 한다.
11. ROS 통신은 후속 현장 통합 단계의 별도 adapter 범위다.
