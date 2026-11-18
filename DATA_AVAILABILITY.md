# 데이터 제공 범위

이 저장소의 데이터 링크는 자료의 구성·용도·SHA-256 및 제공 범위를 설명하는 공개 페이지로 연결된다. 기사 문장 원문, 전체 기사 본문, 사람 검수자·의견·시각, 미사용 후속 Test, 잠긴 평가자료는 배포하지 않는다. 고정 원본 입력은 기존 비공개 보존본에 유지한다.

| 자료 | 구성 및 해시 안내 |
|---|---|
| 훈련 6,000건 | [Train6000](data/train6000.md) |
| Validation 600건 | [Validation600](data/validation600.md) |
| KorSTS 보존 2,000쌍 | [보존 입력](data/retention2000.md) |
| 개발 KorSTS 1,376쌍 | [개발 입력](data/korsts1376.md) |
| 고정 Test 600건 | [Test600](data/test600.md) |
| KLUE-STS519 | [외부 STS519](data/klue519.md) |
| 추가 KLUE-STS 2,000쌍 | [추가 STS2000](data/klue2000.md) |

원본 입력의 전체 식별 정보는 [dataset_manifest.json](metadata/dataset_manifest.json)에 기록되어 있다. 공개 수치만으로 원문 추론과 재학습까지 완전 재현할 수 있다고 주장하지 않는다. 원본 입력이 필요한 연구자는 저자에게 제공 범위와 이용 조건을 확인해야 한다. 이 페이지는 현재 원문 다운로드를 제공하지 않는다.

KorSTS 원출처는 https://github.com/kakaobrain/KorNLUDatasets 이며, KLUE 원출처는 https://github.com/KLUE-benchmark/KLUE 이다. 공개 원자료를 새로 내려받는 것과 본 연구의 고정 부분집합을 확보하는 것은 구분한다. HN은 평가를 위해 의도적으로 바꾼 문장이며 실제 기사 사실로 소개하지 않는다.

Validation과 모든 Test·STS 평가 자료 및 파생 자료의 학습·증류·mining·모델 선택 금지는 공개 후에도 유지한다.
