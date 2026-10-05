# 스키마 v2

길이 m, 질량 kg, 하중 N. `position_m=[x,y,z]`는 상자의 최소 모서리입니다. 회전은 z축 0/90도만 사용합니다. v1 지원 팩의 `allowed_yaw_rad`와 연결할 때 도/라디안 변환이 필요합니다.

## 파일

| 파일 | 레코드 | 용도 |
|---|---|---|
| train/validation/test.episodes.jsonl.gz | 전체 에피소드·정답 순서 | 관측 생성·환경·평가 엔진 |
| train/validation/test.states.jsonl.gz | 관측 중간 상태·후보 표적 | 지도학습·교사 모방 |
| manifest.json | 원본 수·분할·설정·SHA256 | 재현·배포 확인 |
| labels_manifest.json | 상태·후보 수·특징 순서·교사 설정 | 학습 표적 설명 |
| ranker_model.json | 신경망 가중치·학습 정규화·특징 순서 | 추론 시작점 |
| training_report.json | 검증 선택·최종 시험 지표 | 교사 모방 성능 확인 |

## 정책 관측

| 필드 | 의미 |
|---|---|
| pallet | 현재 길이·폭·허용 높이·총무게 한도 |
| placed_boxes | 실제 성공 배치, 지지 부모, 누적 상부 하중 |
| current_box | 측정 지점에서 현재 처리할 박스. 누락된 무게는 null |
| observed_preview | 순서·규격이 확인된 연속 구간. verified_order_rank=1부터 시작 |
| unordered_visible_hints | 가림·누락·낮은 신뢰도 때문에 순서를 확정하지 않는 관측 |
| unseen_inventory | 현재와 확인된 미리보기를 뺀 SKU별 수량. 힌트도 이 수량에 포함 |
| catalog | SKU별 공칭 높이·무게·치수·회전·상부 한도 |
| camera | 시야 길이·관측 상한·측정 sigma·연속성 플래그 |
| robot_feasibility | 이 버전은 항상 NOT_CHECKED |

보이는 후보의 bbox는 0~1 정규화된 합성 검출 위치입니다. 실제 카메라 보정 결과나 촬영 이미지가 아닙니다. `dimension_sigma_m`은 만든 오차 분포의 표준편차이며 모델이 보정한 확률이나 검증된 센서 사양이 아닙니다.

재고의 관계는 다음입니다.

`남은 전체 수량 = 현재 상자 + observed_preview + unseen_inventory`

`unordered_visible_hints`를 이 합에 더하거나 다시 빼면 중복 계산됩니다. 박스가 손상 목적지로 이동해도 남은 전체 수량에서는 빠지지만 적재 성공 개수에는 포함되지 않습니다. 재측정은 입력 인덱스와 재고를 줄이지 않습니다.

## 후보 학습 표적

상태 레코드의 `candidate_targets`에는 후보, 103차원 `features`, `score`, 미래 적재 비율·부피·균형 대용 지표·높이가 들어 있습니다. `best_candidate_index`는 제공된 유효 후보 안에서 교사 점수가 가장 높은 후보입니다. 전체 가능한 배치의 최적해는 아닙니다.

`invalid_probes`는 규칙 차단을 확인할 예시입니다. 신경망 점수 회귀에서는 제외합니다. 모델이 경계·지지 같은 필수 제약을 스스로 배워서 보장할 것으로 가정하지 않습니다.

표적 레코드의 `base_group`, `split`, `scenario`, `camera_profile`은 분할·분석용 메타데이터입니다. 모델 특징에는 넣지 않습니다. `state_id`와 `request_id`도 특징에서 제외합니다.
