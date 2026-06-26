# 출처 및 라이선스 (Attribution)

이 `DINOv2_Gaze/` 폴더는 외부 파운데이션 모델 **DINOv2**와 공개 데이터셋 **Gaze360**을 사용해 시선
추정을 실험한 코드입니다.

> 이 저장소에는 **데이터셋 이미지·학습 가중치를 재배포하지 않습니다.** 우리가 작성한 학습/추론
> 코드만 포함하며, 데이터·가중치는 실행 시 사용자가 직접 받거나 로컬에서 생성합니다.

## 백본: DINOv2

- **모델**: [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) (Meta AI)
- **논문**: *DINOv2: Learning Robust Visual Features without Supervision* (Oquab et al., 2023)
- **라이선스**: Apache-2.0 (가중치는 실행 시 HuggingFace에서 자동 다운로드)

## 학습 데이터: Gaze360

- **데이터셋**: [Gaze360](http://gaze360.csail.mit.edu/) (Kellnhofer et al., ICCV 2019)
- **사용 방식**: cross-person 시선 방향(3D `gaze_dir`) 라벨로 백본 미세조정.
- **취득 경로(HF 미러)**: 이미지 [`immediately/Gaze360-split`](https://huggingface.co/datasets/immediately/Gaze360-split),
  라벨/metadata [`Morning5/Gaze360`](https://huggingface.co/datasets/Morning5/Gaze360).
- ⚠️ **이미지·라벨은 Gaze360 원저작권자 조건을 따른다. 본 저장소는 이미지·파생 `index.csv`를
  포함(재배포)하지 않으며**, `prepare_data.py`로 사용자가 로컬 생성한다. 논문 인용:

```bibtex
@inproceedings{kellnhofer2019gaze360,
  author    = {Kellnhofer, Petr and Recasens, Adria and Stent, Simon and
               Matusik, Wojciech and Torralba, Antonio},
  title     = {Gaze360: Physically Unconstrained Gaze Estimation in the Wild},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```

## 얼굴 검출: InsightFace

- [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l`), 비상업 연구용. 실행 시 자동 다운로드.

## 참고: 3DGazeNet

비교 대상인 3DGazeNet은 sibling 폴더 [`../3DGazeNet_Test/`](../3DGazeNet_Test/)와 그 `SOURCE.md` 참조.

---

작성한 코드(학습/추론/보정 파이프라인) 자체는 개인 연구·실험 목적의 로컬 사용을 전제로 한다.
