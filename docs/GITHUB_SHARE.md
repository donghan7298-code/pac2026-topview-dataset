# 저장소 실행과 공유

기본 브랜치의 `projects/pattern_evaluation`에 점수 평가 프로젝트가 있다. 학습 체크포인트·선택 가중치·설정·검증 보고서와 Python/JSON/HTTP 인터페이스를 함께 관리한다.

```bash
git clone https://github.com/donghan7298-code/pac2026-topview-dataset.git
cd pac2026-topview-dataset
python -m pip install -r requirements.txt
python -m projects.pattern_evaluation plan --input projects/pattern_evaluation/artifacts/sample_observation.json
```

기존 작업 폴더에서는 변경을 커밋하거나 보관한 뒤 `main`의 새 커밋을 가져온다. GitHub Desktop에서도 기본 브랜치를 갱신하면 같은 프로젝트 디렉터리를 확인할 수 있다.

## 배포 구성

- `pacdata`, `configs`, `examples`: 기존 합성 관측 데이터·교사·순차 환경.
- `projects/pattern_evaluation`: 신규 학습·점수 평가·선반 행동 제안·연동 명세.
- `downloads/PAC2026_topview_training_data.zip`: 기존 대용량 데이터 배포본. 저장소 루트에 압축을 풀면 `data/generated`, `data/labels`, `data/model`이 생긴다.
- `projects/pattern_evaluation/artifacts`: 신규 모델·학습/가중치 탐색/비교 기록. 새 실험 출력은 `runs/`에 별도로 저장한다.

합성 데이터는 시드와 설정으로 다시 생성할 수 있다. 서로 다른 모델의 특징 순서와 표적 형식을 섞지 않는다. 관측의 `oracle`는 학습 정책 입력에서 제외하고, 같은 재고 그룹의 카메라 변형은 동일 분할에 둔다.

연동 시에는 저장소 커밋, 모델 파일, 선택 가중치, 단위 m/kg/N, 관측 스키마 `2.0`을 함께 고정한다. 실제 CCTV 인식과 로봇 작업 검증은 [카메라 연동](CAMERA_INTEGRATION.md), [행동 계약](../projects/pattern_evaluation/INTEGRATION.md)을 따른다.
