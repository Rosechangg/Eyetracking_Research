"""DINOv2 gaze 헤드 학습 + 저장 (라이브 배포용).

frozen DINOv2-small 특징 위에 MLP 헤드를 학습해 3D gaze를 출력하고,
헤드 + 특징정규화 통계를 dino_gaze_head.pt로 저장한다.

실행: ../eye_tracking/.venv/Scripts/python.exe train_save.py
"""
import os, sys, csv, math, random
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass
import numpy as np, torch, torch.nn as nn
from collections import defaultdict
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

DEV = "cuda"; CAP = 250; DINO_ID = "facebook/dinov2-small"   # 비교 결과 small이 최선 (large는 더 나쁨)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

rows = list(csv.DictReader(open("index.csv")))
byp = defaultdict(list)
for d in rows: byp[int(d["person"])].append(d)
sel = []
for p, lst in byp.items():
    random.shuffle(lst); sel += lst[:CAP]
random.shuffle(sel)
# 90/10 person split (검증용 sanity)
persons = sorted(byp.keys()); random.shuffle(persons)
val_p = set(persons[:max(1, len(persons)//10)])
print(f"이미지 {len(sel)}장, person {len(persons)}명 (val {len(val_p)}명)")

def unit(v):
    v = np.asarray(v, np.float32); n = np.linalg.norm(v); return v/n if n > 0 else v

dino_proc = AutoImageProcessor.from_pretrained(DINO_ID)
dino = AutoModel.from_pretrained(DINO_ID).to(DEV).eval()

@torch.no_grad()
def extract():
    F, Y, P = [], [], []
    B = 64
    for i in range(0, len(sel), B):
        chunk = sel[i:i+B]
        imgs = [Image.open(d["path"]).convert("RGB") for d in chunk]
        di = dino_proc(images=imgs, return_tensors="pt").to(DEV)
        F.append(dino(**di).pooler_output.cpu().numpy())
        for d in chunk:
            Y.append(unit([float(d["gx"]), float(d["gy"]), float(d["gz"])])); P.append(int(d["person"]))
        if i % (B*40) == 0: print(f"  특징 {i}/{len(sel)}")
    return np.concatenate(F), np.array(Y, np.float32), np.array(P)

print("특징 추출...")
X, Y, P = extract()
tr = np.array([p not in val_p for p in P]); va = ~tr
Xtr = torch.tensor(X[tr]).to(DEV); Ytr = torch.tensor(Y[tr]).to(DEV)
mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
Xtr = (Xtr - mu) / sd
D = X.shape[1]
head = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3)).to(DEV)
opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
print("헤드 학습...")
for ep in range(200):
    head.train(); opt.zero_grad()
    pr = head(Xtr); pr = pr / pr.norm(dim=1, keepdim=True)
    loss = ((pr - Ytr)**2).sum(1).mean(); loss.backward(); opt.step()
# val sanity
head.eval()
with torch.no_grad():
    Xva = (torch.tensor(X[va]).to(DEV) - mu) / sd
    pv = head(Xva); pv = pv / pv.norm(dim=1, keepdim=True)
    ang = torch.acos((pv * torch.tensor(Y[va]).to(DEV)).sum(1).clamp(-1, 1)) * 180/math.pi
print(f"val 각도오차 평균 {ang.mean():.2f}°")

torch.save({"head": head.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(),
            "dino_id": DINO_ID, "feat_dim": D}, "dino_gaze_head.pt")
print("저장: dino_gaze_head.pt")
