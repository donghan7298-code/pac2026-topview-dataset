# pac2026-topview-dataset

PAC 2026 AHEAD 프로젝트 공유 저장소.

이 개발 브랜치에는 이동한 담당 **점수 기반 패턴 평가 v1**을 추가했습니다. 기존 데이터셋 생성 작업과 별도로, 후보별 공간·안정성·하중·미래성과·작업시간·위험을 계산합니다.

- [개발 판단·회의 근거·점수식·선반/PPO 적용·사용법](docs/PATTERN_EVALUATION_KO.md)
- [점수 평가 코드](pattern_evaluation/evaluator.py)
- [태현님 경량 코어와 연결하는 어댑터](pattern_evaluation/core_adapter.py)
- [초기 가중치·정규화 설정](config/pattern_evaluation.json)
- [검증 결과와 현재 한계](experiments/VALIDATION.md)

```bash
python3 -m pattern_evaluation --config config/pattern_evaluation.json
python3 -m unittest discover -s tests -v
```

Python 3.10 이상. 기본 평가기는 추가 의존성 없이 실행됩니다. 코어 연동에는 별도의 `yang8988/pac-mission1` 경량 코어와 NumPy가 필요합니다.

학습된 신경망·물리 선반 시뮬레이터·실제 로봇 검증은 포함하지 않습니다. 별도 회의정리 원문은 조회되지 않아 사용자 메시지의 팀장님 인용 발언과 최종 요약서를 기준으로 구현했습니다. 초기 가중치는 실험 출발값이며, 작은 검증에서 최종 COG 목표 이탈이 있어 자동으로 채택할 프로파일은 선택하지 않았습니다. 기존 플래너 옆에서 점수 기록부터 연결할 수 있습니다.
