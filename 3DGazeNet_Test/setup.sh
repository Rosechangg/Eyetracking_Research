#!/usr/bin/env bash
# 3DGazeNet_Test 셋업: 원본 3DGazeNet(고정 커밋) 클론 + 가중치 다운로드 + venv + 의존성.
# Windows(Git Bash) / Linux 모두 시도. 최신 GPU(RTX 50xx 등)는 시스템에 torch(cu128+)가
# 이미 깔린 상태에서 --system-site-packages 로 재사용하는 것을 권장한다.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PIN=196396c7d00d3bae8fa7ff08b3c79f8286cb5b3a

echo "[1/4] 3DGazeNet 원본 클론 (고정 커밋 $PIN)"
if [ ! -d 3DGazeNet ]; then
  git clone https://github.com/eververas/3DGazeNet.git 3DGazeNet
fi
git -C 3DGazeNet fetch origin "$PIN" 2>/dev/null || true
git -C 3DGazeNet checkout "$PIN" 2>/dev/null || echo "  (고정 커밋 체크아웃 생략 -> 기본 브랜치 사용)"

echo "[2/4] venv 생성 (시스템 패키지 재사용: torch/opencv/onnxruntime)"
python -m venv .venv --system-site-packages
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; GDOWN=.venv/Scripts/gdown.exe
else PY=.venv/bin/python; GDOWN=.venv/bin/gdown; fi
"$PY" -m pip install -U pip
"$PY" -m pip install -r requirements-extra.txt

echo "[3/4] 모델 가중치 다운로드 (저자 Google Drive: checkpoints + eyes3d.pkl)"
"$GDOWN" 1aVbPD51-8EqpJ89TqiTr40pmrpk6iESl -O 3DGazeNet/demo/data_demo.zip
"$PY" -c "import zipfile; zipfile.ZipFile('3DGazeNet/demo/data_demo.zip').extractall('3DGazeNet/demo')"

echo "[4/4] 완료. 실행 예:"
echo "  $PY screen_gaze.py --points 5 --session     # 보정 + 화면 응시점 + 녹화"
echo "  $PY run_zed_gaze.py --session               # 실시간 시선 방향 시각화 + 녹화"
