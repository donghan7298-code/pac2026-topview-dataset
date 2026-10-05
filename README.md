# PAC2026 AHEAD · Top-view Mixed Palletizing Dataset v2

불확정 입고 순서에서 **탑뷰로 실제 확인한 다음 박스**와 **아직 관측하지 못한 재고**를 나눠 사용하는 적재 알고리즘 개발 팩입니다. 데이터 생성 → 순차 적재 → 후보별 미래 평가 표적 생성 → 신경망 점수 학습까지 실행할 수 있습니다.

**합성 개발 데이터입니다.** 주최측 데이터, HD현대로보틱스의 내부 데이터/알고리즘, 실제 CCTV 영상은 포함하지 않습니다. 입력 치수·물성·시간 예산은 개발 가정입니다. 실제 로봇 경로·물리적 안정성은 `NOT_CHECKED`입니다.

## 바로 보기

`examples/topview_viewer.html`을 다운로드해 더블 클릭하면 인터넷 없이 관측 범위·가림·미관측 수량을 확인할 수 있습니다. GitHub에서 HTML 소스만 보이면 파일을 다운로드한 뒤 여세요.

## 포함한 데이터

- **12,000 에피소드 / 770,496 박스 입력**: 서로 다른 기본 재고 구성 250개 × 시나리오 12종 × 카메라 4종.
- 기본 재고별 박스 48~80개, SKU 카탈로그 12종, 치수·무게·하중 한도·회전 제한·팔레트 규격 변형. 20% 그룹은 S 또는 M 반복의 정상 기준 사례, 80%는 12종 혼합 스트레스 사례입니다.
- 카메라 관측 상한 **2 / 4 / 8 / 12개**. 실제 관측량은 시야 길이·상자 간격·가림·검출 누락에 따라 줄어듭니다.
- 학습/검증/시험은 기본 재고 그룹 단위로 분리: **175 / 37 / 38 그룹**, **8,400 / 1,776 / 1,824 에피소드**.
- 별도 `*.states.jsonl.gz`에는 **중간 적재 상태 3,162개 / 유효 후보 학습 표적 23,782개**, 103차원 특징, 미래 탐색 점수, 차단 후보 예시가 들어 있습니다. 정확한 분할별 수는 `labels_manifest.json`을 확인하세요.
- 작은 신경망 학습 코드와 학습 경로를 확인한 예제 모델을 제공합니다. 이 모델은 제한된 교사 탐색의 점수를 따라 배우는 시작점입니다.

같은 상자들은 여러 순서와 카메라 조건에 반복 등장합니다. 770,496개는 독립적인 제품 종류나 실물 촬영 수가 아닌 **에피소드 내 박스 입력의 합**입니다. 학습 표적은 원본 전체의 일부 에피소드에서 생성했습니다.

## Windows에서 실행

Python 3.10 이상을 사용합니다. Visual Studio나 VS Code에서 이 폴더를 열고 터미널을 실행하세요. `python`이 없으면 Windows Python 런처인 `py`로 바꿉니다.

먼저 작은 데이터로 전체 흐름을 확인합니다. 핵심 생성·검증·적재 코드는 외부 패키지가 필요하지 않습니다.

```bash
python -m unittest discover -s tests -v
python -m pacdata.generate --config configs/smoke.json --out data/smoke
python -m pacdata.labels --dataset data/smoke --config configs/labels_smoke.json --out data/labels_smoke
python -m pacdata.validate --dataset data/smoke --labels data/labels_smoke
```

신경망 학습에는 NumPy가 필요합니다.

```bash
python -m pip install -r requirements.txt
python -m pacdata.train_ranker --labels data/labels_smoke --out data/model_smoke --epochs 3
```

이미 생성한 대용량 데이터 ZIP은 **이 저장소 폴더 안에** 압축을 풉니다. 그러면 `data/generated`, `data/labels`, `data/model`이 생깁니다. 기존 예제 모델이 있으므로 새 학습 출력은 다른 폴더에 쓰세요.

```bash
python -m pacdata.train_ranker --labels data/labels --out data/model_new --epochs 20
```

전체를 다시 생성하려면 다음을 실행합니다. 이미 데이터가 있는 출력 폴더에서는 오류로 멈춥니다. 새 폴더 이름을 사용하세요.

```bash
python -m pacdata.generate --config configs/full.json --out data/generated_new
python -m pacdata.labels --dataset data/generated_new --config configs/labels.json --out data/labels_new
python -m pacdata.validate --dataset data/generated_new --labels data/labels_new
```

## 탑뷰를 사용하는 방식

1. 카메라에서 박스의 바닥 길이·폭과 위치를 추정합니다. 높이·무게는 SKU/WMS 또는 깊이 센서·저울 등 별도 정보에서 얻습니다.
2. 한 줄 컨베이어에서 추적된 박스를 적재 지점과 가까운 순서로 정렬합니다. 추월·분기·버퍼가 없는 기본 가정입니다.
3. 추적 이력으로 누락·가림 여부가 확인되고, 순서와 규격의 신뢰도가 충분한 **연속 구간만** `observed_preview`에 넣습니다.
4. 가려진 박스 뒤에 보이는 박스는 `unordered_visible_hints`로 남깁니다. 바로 다음 박스라고 단정하지 않습니다.
5. 현재+확인된 연속 구간을 뺀 나머지는 `unseen_inventory`에 보존합니다. 불확실한 힌트의 수량은 다시 빼지 않습니다.

이 팩은 카메라가 낼 법한 **검출·추적 결과를 수치로 생성**합니다. 실제 검출기나 추적기는 포함하지 않습니다. SKU 식별은 WMS/태그 매칭으로 확인된다는 가정이고, `coverage_complete`는 시간에 따른 추적 시스템이 신뢰할 수 있는 누락 여부를 제공한다는 가정입니다. CCTV 한 장만으로 이를 보장할 수 있다고 주장하지 않습니다. 실제 추적의 연속성을 확인할 수 없다면 해당 구간은 미관측으로 취급합니다.

## 시나리오 12종

| 코드 | 조건 | 학습·시험 목적 |
|---|---|---|
| random | 무작위 입고 | 일반 상태·순서 변화 |
| large_last | 큰 부피 박스 후반 | 미래 공간 확보 |
| heavy_last | 무거운 박스 후반 | 무게 순서·바닥 공간 |
| sku_blocks | 같은 SKU 연속 | 반복 패턴·전환 |
| alternating_size | 큰 것·작은 것 교대 | 빈 공간 분절 |
| tall_flat_mix | 높은·납작한 박스 먼저 | 지지와 높이 활용 |
| heavy_weak_mix | 고밀도·약한 박스 먼저 | 상대 무게·상부 하중 |
| fragile_rotation | 약한 SKU의 회전·하중 제한 강화 | 방향 제약과 하중 차단 |
| damage_route | 일부 손상 박스 | 별도 목적지 분리·재고 갱신 |
| measurement_recheck | 무게 누락·치수 신뢰도 부족 | 재측정 후 재평가 |
| occlusion_gap | 가림·검출 누락 증가 | 불확실한 미리보기 처리 |
| pallet_resize | 처리 10개 후 팔레트 확대 | 새 규격에서 후보 재생성 |

시나리오별 물량 구성은 같은 그룹 안에서 동일합니다. 손상/회전·물성 제한/관측 조건만 필요한 경우 바뀝니다. 기본 팔레트 자체도 여러 규격을 사용합니다. 중간 변경은 확대로 제한했습니다. 이미 적재한 상자가 경계를 벗어나는 축소·재배치는 별도 확장 과제입니다.

## 데이터와 학습 표적을 읽기

```python
from pacdata.io import read_jsonl
from pacdata.packing import PackingEnv, greedy

episode = next(read_jsonl('data/generated/train.episodes.jsonl.gz'))
env = PackingEnv(episode)
observation = env.reset()
candidate = greedy(observation['pallet'], observation['placed_boxes'], observation['current_box'])
observation, reward, done, info = env.step({'type': 'PLACE', 'candidate': candidate})
```

위 예시는 정상 입력의 첫 배치입니다. 실제 반복에서는 손상은 `ROUTE_DAMAGE`, 미확인은 `REMEASURE`, 유효 후보가 없으면 `STOP`으로 분기하세요. `PackingEnv`는 Python 환경이며 Gymnasium의 하위 클래스는 아닙니다.

원본 `episode.oracle.arrivals`는 평가·관측 생성기가 사용하는 정답입니다. **이를 모델 입력으로 사용하면 실제 미관측 미래 순서가 유출됩니다.** 모델에는 `env.reset()`/`env.observation()`의 결과만 전달합니다. 특징 추출은 `pacdata.teacher.features`를 사용하고 episode_id·scenario·split을 특징으로 넣지 않습니다.

학습 표적은 최대 후보 8개, 미래 순서 2개, 깊이 12의 제한된 탐색으로 만듭니다. 관측된 미래는 해당 순서를 사용하고, 미관측 구간은 남은 수량을 보존해 샘플링합니다. 각 미래 단계의 후속 선택은 단순 탐욕 정책입니다. 후보 점수는 예상 미래 적재, 낮은 샘플 성능, 부피, 무게중심 위치 대용 지표, 높이를 조합한 개발용 값입니다. 최적해나 주최측 심사 점수는 아닙니다.

103차원 특징에는 현재 상자·후보·무게중심 대용 지표·재고 요약·최대 12개 관측 박스·5×5 배치 후 heightmap이 들어갑니다. 신경망은 후보 점수를 회귀하고, 유효 후보에서 높은 점수를 선택하는 용도입니다. **규칙 마스크는 학습 모델과 별도로 적용합니다.**

## 검증 범위

자동 검증은 재고 보존, 그룹 분할, 미래 입력 허용 목록, 유효 후보, 누적 하중, 실제 생성한 적재 상태를 확인합니다. 시험에는 미관측 순서를 바꿔도 관측 입력·표적이 바뀌지 않는 경우도 포함합니다. 단일 하부 박스의 100% 지지, 무거운 박스를 가벼운 박스 위에 올리지 않는 규칙을 사용합니다.

학습 보고서의 MSE·교사 후보 일치율·교사 점수 손실은 **교사 모방 지표**입니다. 실제 적재율 향상, 추론 시간 단축, 물리적 안정성을 증명하지 않습니다. 탑뷰 2/4/8/12의 효과는 동일 재고·순서·규칙·계산 예산에서 실제 정책을 실행해 별도로 비교해야 합니다. 신경망으로 줄인 후보는 마지막에 같은 규칙과 필요한 로봇 검증을 통과해야 합니다.

## 팀원 연결 지점

- 알고리즘 담당: `pacdata/packing.py`의 `generate_candidates`, `greedy`를 팀 알고리즘으로 교체. 정책이 받는 관측 형식은 유지.
- 평가·데이터 담당: `mask`, `teacher_score`, 시나리오·물성·예외 처리 조건 검토, 학습·시험 분할과 결과 관리.
- 시각화·로봇 담당: 관측/선택 배치 JSON을 화면·ROS2·RViz·Gazebo/MoveIt에 연결. 가반하중·파지·접근·경로 충돌 검증 추가.

자세한 내용: [데이터 카드](docs/DATA_CARD.md), [입력 스키마](docs/SCHEMA.md), [학습·비교 계획](docs/TRAINING_AND_EVALUATION.md), [GitHub 공유](docs/GITHUB_SHARE.md), [실제 CCTV 연결](docs/CAMERA_INTEGRATION.md).
