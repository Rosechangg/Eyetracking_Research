# DINOv2_Gaze — 파운데이션 백본 기반 시선 추정

자기지도 비전 파운데이션 모델 **DINOv2**를 백본으로 시선을 추정하고, 개인 보정으로 화면 응시점까지
매핑한다. ZED 카메라 프런트엔드(`zed_camera.py`)는 sibling [`3DGazeNet_Test/`](../3DGazeNet_Test/)와 공유.

> 핵심 결론: **frozen 백본은 ~24°에서 막히고, 백본을 미세조정하면 14.6°** (cross-person, Gaze360).
> "큰 백본"이 아니라 **"백본 적응"**이 SOTA의 핵심.

알고리즘 구조·실험 경로·3DGazeNet과의 차이는 **[ALGORITHM.md](ALGORITHM.md)** 참조.

---

## 구성

| 파일 | 역할 |
|------|------|
| `dino_screen_gaze.py` | ★ 9~16점 보정 → **실시간 화면 응시점**(점). fine-tuned 모델 자동 우선 사용 |
| `dino_gaze_live.py` | 라이브 **시선 화살표** 오버레이 |
| `finetune.py` | ★ 백본 **미세조정**(end-to-end) → `dino_gaze_ft.pt` (14.6°) |
| `train_save.py` | frozen 백본 + MLP 헤드 학습·저장 (`dino_gaze_head.pt`) |
| `probe.py` | frozen DINOv2 vs ResNet 특징 비교(파운데이션 백본 효과 증명) |
| `probe_backbones.py` | small/base/large 백본 크기 비교(큰 게 더 나쁨 확인) |
| `prepare_data.py` | Gaze360을 HF에서 받아 `index.csv` 생성(재현용) |
| `zed_camera.py` | ZED OpenCV 스테레오 프런트엔드(동봉) |

---

## 설치 / 실행

```bash
# 의존성 (PyTorch+CUDA, transformers, torchvision, insightface, opencv, huggingface_hub, scipy)
pip install torch torchvision transformers insightface opencv-python huggingface_hub scipy

# 1) 데이터 준비 (Gaze360 일부 + 라벨 → index.csv, gaze360_subset/)
python prepare_data.py

# 2) 백본 미세조정 (→ dino_gaze_ft.pt)
python finetune.py

# 3) 라이브: 16점 보정 후 실시간 화면 응시점 (q/ESC 종료, r 재보정)
python dino_screen_gaze.py
```

> ⚠️ 데이터(`gaze360_subset/`, `index.csv`)와 학습 가중치(`*.pt`)는 **저장소에 포함하지 않는다**
> (용량·라이선스). `prepare_data.py`/`finetune.py`로 로컬 재생성. 출처는 [SOURCE.md](SOURCE.md).

---

## 결과 (cross-person, Gaze360 부분집합)

| 방법 | 각도오차 |
|------|:---:|
| frozen DINOv2-small + MLP | 24° |
| 백본 키우기(large) | 33° (악화) |
| **백본 미세조정** | **14.6°** |

자세한 비교·해석은 [ALGORITHM.md](ALGORITHM.md) §3.
