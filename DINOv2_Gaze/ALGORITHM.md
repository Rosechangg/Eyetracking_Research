# 알고리즘 구성 (DINOv2 시선 추정 파이프라인)

이 폴더는 **자기지도 비전 파운데이션 모델(DINOv2)을 백본으로** 시선을 추정하고, 개인 보정으로
화면 응시점까지 매핑하는 파이프라인이다. 핵심 결론은 **"파운데이션 백본을 동결만 하지 말고
미세조정(fine-tune)해야 SOTA급 정확도가 나온다"**는 것.

---

## 1. 실시간 파이프라인 (라이브 응시점)

```
ZED 스테레오 카메라 (zed_camera.py)
    │  왼쪽 눈 RGB 프레임
    ▼
insightface 얼굴 검출 (buffalo_l)  → 가장 큰 얼굴 bbox
    │  얼굴 크롭(여유 35% 확장 = head crop 근사)
    ▼
DINOv2-small (미세조정됨)  → pooler_output (384d)
    │
    ▼
MLP 헤드 (Linear 384→256, GELU, Dropout, Linear 256→3)
    │  3D 시선 단위벡터 (gx, gy, gz)
    ▼
yaw/pitch 변환  →  16점 Ridge 보정(개인별)  →  화면 응시점 (x, y)
    │
    ▼
화면에 실시간 점 표시 (스무딩 EMA 0.82)
```

- 라이브 스크립트: [`dino_screen_gaze.py`](dino_screen_gaze.py) (보정+응시점), [`dino_gaze_live.py`](dino_gaze_live.py) (시선 화살표)
- 보정은 **DINOv2 출력 → 내 모니터 좌표**를 학습하므로, 학습 도메인(Gaze360)과 ZED의 차이·좌표계
  치우침을 흡수한다. 16점 + Ridge(λ=1.0) + 특징 표준화로 9점 OLS보다 안정적.

---

## 2. 학습 (백본 미세조정)

[`finetune.py`](finetune.py) — frozen이 아니라 백본+헤드를 **end-to-end** 학습.

| 항목 | 값 |
|------|----|
| 백본 | DINOv2-small (동결 해제) |
| 데이터 | Gaze360 부분집합 (사람별 ≤250장, 약 16k장) |
| 라벨 | `gaze_dir` 3D 시선벡터 (Gaze360 metadata) |
| 학습률 | 차등: 백본 1e-5 / 헤드 1e-3 (AdamW, 코사인) |
| 증강 | 수평 플립(+ gx 부호 반전) |
| 손실 | 코사인 손실 `1 - cos(pred, target)` |
| 분할 | person 단위 70/30 (cross-person) |
| 결과 | **cross-person 14.6°** |

데이터 준비는 [`prepare_data.py`](prepare_data.py) 참조 (Gaze360을 HF에서 받아 `index.csv` 생성).

---

## 3. 실험 경로에서 얻은 핵심 발견

같은 입력(얼굴 이미지)으로 "VLM/파운데이션을 어떻게 쓰는가"를 바꿔가며 측정한 결과:

| 방법 | 핵심 | cross-person 각도오차 | 스크립트 |
|------|------|:---:|---|
| 생성형 VLM에게 직접 질의 | 챗봇에게 "어디 봐?" | 사실상 실패(화면칸 0%) | (3DGazeNet_Test의 `vlm_gaze`) |
| frozen 파운데이션 백본 + MLP | 특징만 빌려 헤드만 학습 | 21~24° | `probe.py`, `train_save.py` |
| 백본 키우기 (small→base→large) | 큰 게 답이 아님 | 24 → 26 → 33° (악화) | `probe_backbones.py` |
| **백본 미세조정 (fine-tune)** | 백본을 gaze에 적응 | **14.6°** | `finetune.py` |

**교훈**: "큰 백본"이 아니라 **"백본을 적응(미세조정)시키는 것"**이 SOTA의 핵심.
동결 probe는 ~24°에서 천장(파운데이션 특징은 클수록 추상적이 되어 저수준 시선 신호가 희석됨).

---

## 4. 3DGazeNet과의 차이

| | 3DGazeNet (sibling 폴더) | 이 폴더 (DINOv2 fine-tuned) |
|---|---|---|
| 백본 | gaze 전용 CNN (ResNet/MobileViT) | **DINOv2** (자기지도 파운데이션, LVD-142M) |
| 사전학습 | gaze 데이터(ETH-XGaze/GazeCapture/Gaze360/MPIIFaceGaze) | 라벨 없는 대규모 이미지(자기지도) → gaze 미세조정 |
| 출력 | 3D eye mesh + gaze 벡터 | 3D gaze 벡터 |
| 얼굴 검출 | insightface | insightface (동일) |
| 화면 매핑 | 9~25점 ridge 보정 | 16점 ridge 보정 |
| Gaze360 오차(참고) | 8.8° (전체 데이터 학습) | 14.6° (부분 데이터·6 epoch) |
| 성격 | 도메인 매칭·정밀, 즉시 사용 | 파운데이션 백본을 적은 코드로 SOTA 방식(미세조정)으로 체험 |

**한 줄 요약**: 3DGazeNet은 "처음부터 gaze 전용으로 만든 정밀 모델", 이 폴더는 "범용 자기지도 백본
(DINOv2)을 gaze에 미세조정한 모델". 둘 다 *"백본 + 헤드 + 보정"*이라는 같은 뼈대지만, **백본의
출신(전용 학습 vs 자기지도 파운데이션)이 다르다.** 최신 SOTA(UniGaze, Gaze-LLE 등)는 후자 계열로 이동 중.
