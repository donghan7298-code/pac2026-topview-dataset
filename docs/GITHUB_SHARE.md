# GitHub로 팀원과 공유

## 파일 역할

- `PAC2026_topview_github_ready.zip`: 저장소에 올릴 코드·설정·문서·작은 예제·오프라인 HTML.
- `PAC2026_topview_training_data.zip`: 생성된 원본 데이터·학습 표적·예제 모델·검증 보고서. 저장소 폴더 안에 압축을 풉니다.

대용량 출력은 `.gitignore`에 포함되어 코드와 샘플만 Git에 올라갑니다. 원본을 다시 생성할 수 있고 manifest의 SHA256으로 같은 배포본인지 확인할 수 있습니다. 팀에는 코드 저장소와 데이터 ZIP을 함께 공유하세요. GitHub Releases의 첨부 파일로 데이터 ZIP을 올리는 방식도 사용할 수 있습니다. 이 패키지 자체가 원격 저장소나 Release를 이미 생성한 것은 아닙니다.

## 가장 쉬운 방법: GitHub Desktop

1. 코드 ZIP을 압축 해제합니다.
2. GitHub Desktop에서 **File → Add local repository**로 이 폴더를 선택합니다. 저장소가 아니라는 안내가 나오면 **create a repository here**로 현재 폴더를 저장소로 만듭니다.
3. Changes에서 README, pacdata, configs, docs, tests, examples 등이 포함되고 대용량 data 폴더는 제외됐는지 확인합니다.
4. 초기 커밋을 만들고 **Publish repository**를 누릅니다. 팀 공유용으로 우선 Private를 선택하세요.
5. 저장소 Settings → Collaborators에서 팀원이 접근할 수 있도록 설정합니다. 저장소 URL을 팀원에게 공유합니다.
6. 데이터 ZIP도 함께 공유하거나 Release 첨부로 업로드합니다. 팀원은 같은 폴더 구조로 압축을 풀고 README의 smoke 명령부터 실행합니다.

공유·초대는 사용자가 수행하는 안내이며 자동으로 초대장을 보낸 작업은 없습니다.

## 터미널로 올리기

Git과 GitHub CLI가 설치되어 있고 명령줄 인증이 되어 있다면, 코드 폴더에서 실행합니다. 처음 사용하는 계정은 `gh auth login`의 안내를 따르세요. Git의 이름·이메일은 본인 계정 정보를 설정합니다.

```bash
git init -b main
git add .
git commit -m "Add synthetic topview packing dataset toolkit"
gh repo create pac2026-topview-dataset --private --source=. --remote=origin --push
```

이미 만들어 둔 **빈 저장소**에 올릴 경우에는 마지막 줄 대신 본인 저장소 주소를 사용합니다.

```bash
git remote add origin https://github.com/OWNER/pac2026-topview-dataset.git
git push -u origin main
```

`OWNER`는 본인/팀 GitHub 계정명으로 바꿉니다. 기존 README나 작업 내용이 있는 저장소에서는 새 폴더/브랜치로 가져와 변경을 확인하세요. 강제 push는 이 안내에 포함하지 않습니다.

## 팀원에게 전달할 내용

“탑뷰 관측 상한 2/4/8/12개와 가림·오차를 포함한 합성 적재 데이터입니다. 입고 목록 외에 적재 중간 상태와 후보별 미래 평가 표적도 있습니다. README의 smoke 명령으로 먼저 실행해 주세요. 알고리즘은 observed_preview와 unseen_inventory를 입력으로 사용하고, oracle의 실제 미래 순서는 넣지 않습니다. 로봇 검증과 실제 CCTV 인식은 별도 연결 항목입니다.”

공식 안내: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
