# 출처 및 라이선스 (Attribution)

이 `3DGazeNet_Test/` 폴더는 외부 연구 모델 **3DGazeNet**을 ZED 카메라와 연동·실험한 코드입니다.

> 이 저장소에는 **3DGazeNet 원본 코드와 가중치를 재배포하지 않습니다.** 우리가 작성한 연동 코드
> + 설치 스크립트(`setup.sh`)만 포함하며, 원본은 실행 시 사용자가 직접 내려받습니다(아래 참조).

## 핵심 알고리즘: 3DGazeNet

- **프로젝트**: [eververas/3DGazeNet](https://github.com/eververas/3DGazeNet)
- **논문**: *3DGazeNet: Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views* (ECCV 2024)
- **저자**: Evangelos Ververas, Polydefkis Gkagkos, Jiankang Deng, Michail Christos Doukas, Jia Guo, Stefanos Zafeiriou
- **고정 커밋**: `196396c7d00d3bae8fa7ff08b3c79f8286cb5b3a` (2025-01-21, `main`)
- **모델 가중치**: 저자가 Google Drive로 배포 ([file id `1aVbPD51-8EqpJ89TqiTr40pmrpk6iESl`](https://drive.google.com/file/d/1aVbPD51-8EqpJ89TqiTr40pmrpk6iESl/view)) → `setup.sh`가 `3DGazeNet/demo/data/`로 내려받음

> ### ⚠️ 라이선스 주의
> 원본 레포 `eververas/3DGazeNet`에는 **명시된 라이선스(LICENSE 파일)가 없습니다**(GitHub 라이선스: 없음).
> 라이선스가 없으면 기본적으로 **저작권자 권리 보유(all rights reserved)** 이므로, 본 저장소는 원본
> 코드/가중치를 **포함(재배포)하지 않고** 링크와 설치 스크립트로만 연결합니다. 개인 연구/실험 목적의
> 로컬 사용으로 한정하며, 사용 전 원저자의 조건을 확인하세요. 논문 인용 시 아래 BibTeX 사용.

```bibtex
@inproceedings{ververas20253dgazenet,
  author    = {Ververas, Evangelos and Gkagkos, Polydefkis and Deng, Jiankang and
               Doukas, Michail Christos and Guo, Jia and Zafeiriou, Stefanos},
  title     = {3DGazeNet: Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```

## 얼굴 검출: InsightFace
- [deepinsight/insightface](https://github.com/deepinsight/insightface) (License: **MIT**). `buffalo_l`(SCRFD) 모델은 최초 실행 시 `~/.insightface/models/`로 자동 다운로드.

## (선택) VLM 비교: Google Gemini
- `vlm_gaze.py`는 3DGazeNet(정밀 수치) vs **Gemini VLM**(자연어 추정) 비교 데모. `GEMINI_API_KEY` 환경변수 필요(키는 코드에 포함되지 않음). 얼굴 이미지가 외부(Google)로 전송되므로 동의 후 사용.

## 이 폴더에서 새로 작성한 코드 (연동 글루 / 원본 무수정)

| 파일 | 역할 |
|------|------|
| `zed_camera.py` | ZED를 OpenCV 스테레오 USB 스트림으로 받아 왼쪽 눈 RGB만 추출 (depth 미사용; pyzed 있으면 그 경로도 지원) |
| `run_zed_gaze.py` | ZED 프레임 → 3DGazeNet 추론 → 실시간 시선 시각화 + 세션 녹화(mp4)/gaze 데이터(tsv) |
| `screen_gaze.py` | 모니터 캘리브레이션(시선→화면 좌표 회귀) + 화면 응시점 시각화 + 녹화/PiP |
| `vlm_gaze.py` | 3DGazeNet vs Gemini VLM 시선 추정 비교 데모 |
| `setup.sh` | 원본 3DGazeNet(고정 커밋) 클론 + 가중치 다운로드 + venv 구성 |
