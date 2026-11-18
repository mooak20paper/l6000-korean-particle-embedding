# 한국어 조사 변화에 따른 의미 변별력 향상을 위한 BGE-M3 문장 임베딩 미세조정 연구

L6000 논문의 공개 재현자료 저장소이다. 최종 LoRA 어댑터, 실제 사용한 코드·설정, 수치 결과와 통계, 실행 환경, 자료 식별 해시를 제공한다. 기존 비공개 연구 저장소와 별도의 새 이력으로 공개하였다.

## 자료 찾기

| 자료 | 공개 위치 |
|---|---|
| 최종 모델 | [LoRA 어댑터](model/) |
| 훈련·평가 코드 | [code](code/) |
| L6000 설정 | [실제 설정](configs/L6000_run_config.json) |
| Test 전 선정 규칙 | [선정 규칙](configs/selection_rule.json), [고정 선정 결과](results/selected_models/selection.json) |
| 학습량·보존 가중치 개발 비교 | [9조건 개발 결과](results/development/scale_comparison.json) |
| Test600·KLUE-STS519 | [모델 비교](results/test519/paper_comparison.csv), [통계](results/test519/paired_statistics.json) |
| 추가 KLUE-STS 2,000쌍 | [모델 비교](results/sts2000/paper_comparison.csv), [통계](results/sts2000/paired_statistics.json) |
| 추가 학습 seed | [결과](results/seeds/summary.json) |
| 실행 환경 | [environment](environment/) |
| 데이터 구성·해시·제공 범위 | [데이터 안내](DATA_AVAILABILITY.md), [데이터별 안내](data/) |
| 파일 무결성 | [MANIFEST.json](MANIFEST.json) |

## 핵심 결과

| 평가 | BGE-M3 | L6000 |
|---|---:|---:|
| Test600 정확도 | 43.50% | 95.00% |
| KLUE-STS519 Spearman | 0.8773 | 0.8337 |
| 추가 KLUE-STS 2,000쌍 Spearman | 0.9207 | 0.8961 |

519쌍의 상대 하락률은 4.97%, 추가 2,000쌍에서는 2.66%이다. 모든 추가 seed가 5% 이내인 것은 아니다. 기존 Test의 반복 관찰과 공개 학습 분할에서 선정한 추가 2,000쌍을 독립 확증으로 간주하지 않는다. 조사 변별이 중요한 도메인의 선택적 활용 가능성을 다루며, 범용 모델의 전면 대체나 실제 RAG 검색 성능 향상을 입증한 결과는 아니다.

## 재현 및 공개 범위

`python verify_release.py`는 파일 해시와 기존 수치의 무결성만 확인하며 모델을 불러오거나 새 실험을 실행하지 않는다. [재현 절차](REPRODUCING.md)를 따른다. 문장 원문과 사람 검수 기록은 포함하지 않으므로 원문 추론·재학습의 완전 재현에는 별도 접근이 필요하다.

공개 라이선스를 별도로 부여하지 않았으며, [이용 조건](THIRD_PARTY_NOTICES.md)과 각 원자료의 조건을 확인한다. 교수 메일·논문 초안·검수 앱 DB·미사용 평가자료·인증정보와 기존 저장소의 Git 이력은 포함하지 않는다.
