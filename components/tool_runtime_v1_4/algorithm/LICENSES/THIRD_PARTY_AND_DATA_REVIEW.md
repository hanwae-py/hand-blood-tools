# 외부 전송 전 라이선스·데이터 검토

이 release candidate는 법률 검토 완료를 의미하지 않는다. 최종 전송 전 다음을 확인한다.

- RF-DETR source/package 및 base weights의 적용 라이선스와 고지 의무
- PyTorch, torchvision, OpenCV, NumPy의 라이선스 고지
- fine-tuned checkpoint의 소유권과 외부 기관 제공 가능 범위
- 학습·검증 영상 및 annotation의 IRB/데이터 이용·반출 범위
- 문서·예제에 임상 원본 또는 식별 정보가 포함되지 않았는지

본 bundle에는 임상 원본 영상, 전체 dataset, annotation 또는 training optimizer state를 넣지 않는다.

