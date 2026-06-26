"""Gaze360 부분집합을 HF에서 받아 학습용 index.csv + gaze360_subset/ 생성 (재현용).

저장소엔 데이터를 포함하지 않으므로 로컬에서 1회 실행한다.
- 이미지: immediately/Gaze360-split (part aa ~5GB만 사용 → 약 48k장/105명)
- 라벨:   metadata.mat의 gaze_dir(3D) + person_identity + recording/frame
- 매핑:   imgs/rec_{recording:03d}/head/{person:06d}/{frame:06d}.jpg

실행: python prepare_data.py
주의: Gaze360 라이선스를 준수할 것 (SOURCE.md 참조).
"""
import os, tarfile, csv
import scipy.io as sio
from huggingface_hub import hf_hub_download

os.makedirs("gaze360_subset", exist_ok=True)

print("part aa(~5GB) 다운로드...")
pa = hf_hub_download("immediately/Gaze360-split", "Gaze360.tar.partaa", repo_type="dataset")

print("스트리밍 추출(jpg + metadata.mat)...")
n = 0
try:
    with tarfile.open(name=pa, mode="r|") as tf:        # 분할 tar이라 마지막에 EOF 에러 정상
        for m in tf:
            if m.isfile() and (m.name.endswith(".jpg") or m.name.endswith(".mat")):
                tf.extract(m, "gaze360_subset"); n += 1
except Exception as e:
    print("스트림 끝:", str(e)[:40])
print(f"추출 {n} 파일")

print("metadata.mat → index.csv 매핑...")
m = sio.loadmat("gaze360_subset/metadata.mat")
recording, frame = m["recording"][0], m["frame"][0]
person, gaze = m["person_identity"][0], m["gaze_dir"]
rows = []
for i in range(len(recording)):
    p = f"gaze360_subset/imgs/rec_{recording[i]:03d}/head/{person[i]:06d}/{frame[i]:06d}.jpg"
    if os.path.exists(p):
        rows.append((p, float(gaze[i, 0]), float(gaze[i, 1]), float(gaze[i, 2]),
                     int(person[i]), int(recording[i])))
with open("index.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "gx", "gy", "gz", "person", "rec"]); w.writerows(rows)
print(f"완료: index.csv ({len(rows)} 이미지, {len(set(r[4] for r in rows))} 명)")
print("→ 이제 finetune.py 실행")
