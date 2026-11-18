# 실행 환경

training.json은 원래 L6000 학습의 실제 기록이다. 학습은 Tesla T4, torch 2.11.0+cu128 및 FP16 autocast, 평가는 T4, torch 2.11.0+cu130 및 FP32였다. requirements 파일은 패키지 버전을 기록하며 GPU 드라이버와 CUDA까지 자동 재현하는 lockfile은 아니다. CUDA 빌드는 해당 JSON과 일치시켜야 한다. 학습 코드의 환경 assert를 해제해 다른 환경을 동일한 재현으로 보고하지 않는다.

모든 모델을 .eval()과 inference_mode()로 평가한다. 기본 및 한국어 파생 모델은 revision을 고정한다. upskyy/bge-m3-korean은 mean pooling, 나머지는 CLS pooling을 사용한다. 최대 160토큰, 절단 금지, L2 정규화와 FP64 cosine, 동률 허용 오차 1e-6을 따른다. 학습과 평가 환경은 서로 다른 기록이다.
