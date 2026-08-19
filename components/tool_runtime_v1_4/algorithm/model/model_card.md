# CAM4 RF-DETRSegSmall model card

## 식별

- model version: `cam4-rfdetr-seg-small-v1`
- architecture: `RFDETRSegSmall`
- frozen class count: 8
- checkpoint file: `cam4_rfdetr_seg_small_v1.pth`
- checkpoint SHA256: `fff1e97a0259f11597ef6c5cf70506d1ec40eafbb5f391a498b6bee452c1fe6f`
- current operating threshold: `0.5`
- validated checkpoint input color order: `BGR`

현재 checkpoint의 standalone pose reference와 realtime benchmark는 OpenCV BGR 입력 경로로
수행되었으므로 v1은 재현성을 위해 BGR을 checkpoint contract로 고정한다. API 호출자는 자신의
입력이 RGB인지 BGR인지 명시하며 adapter가 내부 BGR로 변환한다. training summary의 9-instance
single-frame sanity와 pose frame-99의 11-instance reference는 서로 다른 입력 sample이다.

## class 순서

모델 index는 0..7이고 canonical ID는 1..8이다. 상세 이름과 alias는 `ontology.json`이 정본이다.

## 학습 및 검증 범위

- CAM4 0704_5 teacher COCO segmentation v2
- train 593 frames / 4,800 instances
- validation 100 frames / 981 instances
- 30 epochs, selected epoch 26, resolution 384
- validation bbox mAP50-95: 0.71397
- validation mask mAP50-95: 0.65240
- validation mask mAP50: 0.92692

위 수치는 temporal holdout teacher-consistency이며 독립적인 dense manual ground truth 성능이 아니다.
tray 실촬영 데이터에 대한 성능 수치가 아니다.

## 알려진 제한

- CAM4 자료만으로 학습 및 현재 검증됨
- tray는 실제 demo 대상이지만 촬영·독립 검증 자료가 아직 없음
- occlusion, glare, hand overlap, cable 포함 오류, 미등록 기구는 별도 평가가 필요
- class ID는 checkpoint index가 아니라 `ontology.json`의 canonical ID로 교환해야 함

## 사용 제한

임상 판단을 자동화하거나 pose validity 확인 없이 로봇 동작 명령으로 직접 연결하지 않는다.
학습 데이터와 base checkpoint를 포함한 제3자/데이터 라이선스 및 반출 승인은 final 외부 전송 전에
기관 담당자가 확인해야 한다.
