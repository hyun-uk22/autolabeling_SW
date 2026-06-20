# Windows 데스크톱 앱 및 Setup 명세

## 실행

개발 환경에서 GUI를 실행한다.

```powershell
.\.venv\Scripts\python.exe desktop_app.py
```

installer를 실행하기 전에 브라우저 인터페이스로 workflow를 확인할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m streamlit run web_app.py
```

접속 주소는 `http://localhost:8501`이다. Streamlit과 데스크톱 앱은 같은 사용자 `.env`를 사용한다.

화면은 형식 변환, 라벨 생성, 평가, 설정으로 구성된다. 작업은 기존 LangGraph `WorkflowPlan`으로 변환되어 CLI와 동일한 runtime에서 실행된다. 라벨 생성 실행 전에는 고비용 모델 호출 가능성을 확인하는 승인 창이 표시된다.

앱 시작 시 workspace 선택 창이 항상 표시되고 마지막 선택 경로가 기본값으로 제공된다. `표준 폴더 구조 생성`을 선택한 경우 다음 경로를 만들며, 화면에서도 workspace 상대 기본값으로 사용한다.

```text
data/raw
data/labeled
data/visualized
data/converted
data/ground_truth
data/reports
configs/plugins.json
```

실행 계획을 만들 때 상대 경로를 workspace 아래 절대 경로로 변환한다. workspace 내부 파일을 파일 선택기로 선택하면 가능한 경우 다시 상대 경로로 표시한다. 선택값은 `%APPDATA%\AutoLabel\workspace.json`에 저장한다.

## 사용자 설정

설정 화면은 다음 값을 관리한다.

- `AWS_REGION`, `AWS_PROFILE`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `LOW_MODEL`, `HIGH_MODEL`, `PLANNER_MODEL`

저장 위치:

```text
%APPDATA%\AutoLabel\.env
```

installer와 Git 저장소에는 실제 `.env`를 포함하지 않는다. 설정 파일은 Windows 사용자 프로필에 평문으로 저장되므로 공유 계정에서는 별도의 운영 보안 정책이 필요하다.

## 빌드

필수 도구:

- 프로젝트 `.venv`
- `requirements.txt`
- `requirements-desktop.txt`
- Inno Setup 6

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\packaging\build.ps1
```

산출물:

```text
dist\AutoLabel\AutoLabel.exe
dist-installer\AutoLabel-Setup.exe
```

`build.ps1 -SkipInstaller`는 PyInstaller 애플리케이션 번들만 생성한다.

`build/`, `dist/`, `dist-installer/`는 Git에서 제외한다. 공개 installer는 코드 서명과 clean VM 검증을 마친 뒤 GitHub Releases에 별도 첨부하며 저장소 commit에는 포함하지 않는다.

## 설치 정책

- 사용자 권한으로 `%LOCALAPPDATA%\Programs\AutoLabel`에 설치한다.
- 시작 메뉴 바로 가기를 생성한다.
- 바탕 화면 바로 가기는 installer에서 선택할 수 있다.
- README, 변환·생성 명세, plugin 설정 예시를 설치 디렉터리에 포함한다.
- uninstall entry를 등록한다.

## Specialist 모델

기본 installer는 크기와 CUDA 환경 의존성을 제한하기 위해 다음 선택 패키지를 제외한다.

```text
torch
transformers
ultralytics
easyocr
```

따라서 기본 installer에서는 API 기반 VLM 생성, 라벨 변환, 평가를 사용할 수 있다. Grounding DINO, SAM, Ultralytics pose/tracking, EasyOCR plugin을 포함한 별도 GPU/CPU 배포판은 대상 CUDA/PyTorch 조합을 고정해 별도로 빌드해야 한다. 모델 가중치는 installer에 포함하지 않고 plugin 최초 실행 시 공급자 cache에 받는 방식을 사용한다.

## 배포 전 점검

- 신뢰할 수 있는 코드 서명 인증서로 installer와 실행 파일에 서명
- 깨끗한 Windows VM에서 설치·실행·제거 검사
- 실제 사용하는 API provider별 인증 검사
- 대상 CPU/GPU 환경별 specialist 배포판 분리
- 배포 버전에 맞게 `installer.iss`의 `MyAppVersion` 갱신
