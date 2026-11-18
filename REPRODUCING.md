# 재현 절차

## 모델을 실행하지 않는 확인

`python verify_release.py`로 명시된 파일과 해시, 어댑터 식별자, 수치 파일의 문항 수와 정답 수를 확인한다. 기존 결과를 확인하는 명령이며 새로운 실험이 아니다.

## 원본 입력을 확보한 뒤의 학습

L6000의 원래 입력 패키지는 `configs/training_input_manifest.json`의 19개 파일을 포함한다. 연구자에게 원본 입력 접근 권한을 받아 동일 manifest의 파일을 준비한다. `configs/L6000_run_config.json`을 `run_config.json`으로, `code/training/`의 두 Python 파일을 입력 패키지의 같은 이름으로 둔다. 다른 입력은 해시가 같은 원본을 그대로 사용한다. 누락·변경 파일을 임의로 생성해 guard 통과로 표시하지 않는다.

```text
python code/training/train_retention.py --package /path/to/original_package --preflight-only
python code/training/train_retention.py --package /path/to/original_package --output /path/to/new_run
```

실제 학습은 저장된 GPU·패키지 조건을 맞춘 뒤 명시적으로만 실행한다. 1 epoch / seed 17 / r16 / alpha32 / dropout0.1 / lr1e-4 / temperature0.05 / batch4 / 조건 C / lambda10이다. 조사 대조 손실과 KorSTS 2,000쌍의 고정 교사 cosine 보존 MSE를 합한다. Test와 KLUE-STS를 학습 패키지에 넣지 않는다. 학습 코드의 assert는 원래 입력·환경을 확인하기 위한 것으로 해제하지 않는다.

## GPU 평가

평가 원본 입력과 reference 점수를 확보한 뒤 `configs/test519_protocol.json`을 `protocol_v2.json`으로 복사한 bundle을 구성한다. 프로토콜 `input_files`의 고정 파일과 `model/`의 두 어댑터 파일을 `adapter/`에 둔다. `run_colab_evaluation.py`와 `analyze_results.py`를 bundle 루트에 둔 다음, bundle의 파일별 SHA-256을 `bundle_manifest.json` 사전으로 기록한다. 참조 파일이 없으면 원래 재현 관문을 통과한 실행으로 보고하지 않는다.

```text
python code/evaluation/run_colab_evaluation.py --bundle /path/to/test519_bundle --output /path/to/new_evaluation
```

추가 2,000쌍은 `configs/sts2000_protocol.json`을 `protocol.json`으로 둔 별도 bundle로 평가한다. `klue2000.json`, `klue519.json`, 기존 5개 모델의 519쌍 참조 점수, `adapter/`, 원래 `preflight.json`과 명시 manifest를 포함한다.

```text
python code/evaluation/run_sts2000.py --bundle /path/to/sts2000_bundle --output /path/to/new_sts2000_evaluation
```

기존 519쌍 재현 확인과 2,000쌍의 주 평가는 분리하여 보고한다. 원문을 확보하지 못했거나 reference 또는 guard가 누락된 실행을 완전 재현으로 표시하지 않는다.

## 결과 읽기

문항별 CSV의 `cos_ap - cos_ahn > 1e-6`을 정답으로 판단한다. 동일 문항과 STS 문장쌍을 모델 간 함께 재표집한다. 기존 주요 비교는 부록의 7개 비교에 Holm 보정을 적용했으며, 한국어 파생 5모델 전체 비교는 10개 비교에 Holm 보정했다. 두 검정 집합을 섞지 않는다. `code/analysis/`의 원래 분석 스크립트에는 당시 연구 폴더 구조가 필요하므로, 별도 소스 이력의 재현 자료이며 이 폴더에서 단독 실행되는 일반 CLI가 아니다.
