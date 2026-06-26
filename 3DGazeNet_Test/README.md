# 3DGazeNet_Test — ZED 카메라 + 3DGazeNet 시선 추정

[eververas/3DGazeNet](https://github.com/eververas/3DGazeNet) (ECCV 2024)을 **ZED 카메라**와 연동해
실시간으로 3D 시선(gaze)을 추정하고, 모니터 캘리브레이션으로 **화면 응시점(Point-of-Regard)** 까지
시각화·녹화한다.

> 출처/라이선스: **[SOURCE.md](SOURCE.md)** 참고. 원본 3DGazeNet은 라이선스 미명시라 이 저장소엔
> **원본 코드/가중치를 포함하지 않고** 설치 스크립트로만 연결한다(개인 연구용).

---

## 구성

| 파일 | 역할 |
|------|------|
| `zed_camera.py` | ZED를 OpenCV 스테레오 USB 스트림으로 받아 왼쪽 눈 RGB만 추출 (ZED SDK 불필요) |
| `run_zed_gaze.py` | 실시간 시선 *방향* 시각화 + 세션 녹화(mp4) + gaze 데이터(tsv) |
| `screen_gaze.py` | 모니터 캘리브레이션 + 화면 *응시점* 시각화 + 녹화 + 내 영상 PiP |
| `vlm_gaze.py` | 3DGazeNet(정밀 수치) vs Google Gemini VLM(자연어 추정) 비교 데모 |
| `setup.sh` | 원본 3DGazeNet(고정 커밋) 클론 + 가중치 다운로드 + venv |

---

## 설치

전제: **Python + PyTorch(CUDA)** 가 설치돼 있어야 한다. 최신 GPU(RTX 50xx 등)는 cu128+ 빌드 필요.
`setup.sh`는 그 위에서 `--system-site-packages` venv로 torch를 재사용하고 나머지만 설치한다.

```bash
cd 3DGazeNet_Test
bash setup.sh
```

`setup.sh`가 하는 일:
1. 원본 [3DGazeNet](https://github.com/eververas/3DGazeNet) 클론 (고정 커밋 `196396c`)
2. `--system-site-packages` venv 생성 + `requirements-extra.txt` 설치 (insightface, gdown 등)
3. 저자 Google Drive에서 가중치(`checkpoints/*.pth`, `eyes3d.pkl`)를 `3DGazeNet/demo/data/`로 다운로드

> 얼굴 검출 모델(`buffalo_l`)은 최초 실행 시 insightface가 `~/.insightface/`로 자동 다운로드.

---

## 사용법

```bash
# (Windows venv 기준 파이썬 경로 예: .venv/Scripts/python.exe)
PY=.venv/Scripts/python.exe

# 1) 실시간 시선 방향 + 녹화
$PY run_zed_gaze.py --session

# 2) 모니터 캘리브레이션 -> 화면 응시점 (메인). 매번 처음부터 새로 보정(권장: 가장 정확)
$PY screen_gaze.py --points 5
#    + 녹화/내 영상 PiP까지 저장하려면 --session (단 FPS가 내려가 지터가 늘 수 있음)
$PY screen_gaze.py --points 5 --session

# 3) 동작 확인(이미지 1장 스모크 테스트)
$PY run_zed_gaze.py --image 3DGazeNet/demo/data/test_images/img1.jpg

# 4) (선택) 3DGazeNet vs Gemini VLM 비교  (GEMINI_API_KEY 필요)
$PY vlm_gaze.py --image 3DGazeNet/demo/data/test_images/img1.jpg
```

### screen_gaze.py (화면 응시점 + 녹화)
- **보정**: 전체화면 타겟을 하나씩 응시하며 `SPACE`. 끝나면 시선→화면 좌표 **ridge 정규화 회귀** 적합
  (One-Euro 필터로 지터 억제, 머리이동 보정 특징, LOO-CV 정확도 출력). 점이 많을수록 정확(`--points 5`).
- **항상 새 보정이 기본**: 실행할 때마다 처음부터 보정한다. 저장된 보정은 `--load`를 명시할 때만
  재사용하며, 머리 위치가 보정 때와 달라지면 어긋나므로 "안 맞으면" `--load` 없이 그냥 다시 보정하면 된다.
- **키**: `SPACE`=캡처, `r`=재보정(언제든 처음부터), `g`=타겟토글, `q`/`ESC`=종료.
- **녹화**: `--session` 이면 `recordings/session_<시각>/` 에 `screen.mp4`(시각화 + 내 영상 PiP 하단중앙) +
  `gaze.tsv` 저장. `q`/`ESC` 종료 시 마무리. 콘솔에 `[rec] ... 녹화됩니다 -> ...` 가 보이면 ON.
- 주요 옵션: `--load`(보정 재사용), `--recalibrate`(강제 재보정), `--no-mirror`(PiP 거울 끄기),
  `--pip-width N`, `--record-scale`, `--no-pip`, `--no-display`, `--max-frames N`.

---

## 알아둘 점 / 한계
- 3DGazeNet은 **단안 RGB** 시선 추정기라 ZED **depth는 사용하지 않는다**(ZED를 일반 웹캠처럼 사용).
- 화면 응시점은 시선 *방향*을 보정 회귀로 화면 좌표에 매핑하므로 **머리를 (대략) 고정**한 상태가 가장 정확.
  머리를 크게 움직이면 정확도가 떨어진다 → `r`로 재보정. (머리 움직임 강건/수직 정확도는 ZED depth 기반
  3D ray-cast가 필요하며 향후 과제.)
- 수직(pitch) 시선은 모델 특성상 수평(yaw)보다 약할 수 있다(눈꺼풀 가림 등).
