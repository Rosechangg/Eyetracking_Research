"""백본 3종(small/base/large) 공정 비교 — 개선된 레시피.

frozen 백본 특징을 L2 정규화 후 동일 헤드로 학습(cross-person). 어느 크기가 실제로 최선인지 확인.
실행: ../eye_tracking/.venv/Scripts/python.exe probe_backbones.py
"""
import os, sys, csv, math, random
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass
import numpy as np, torch, torch.nn as nn
from collections import defaultdict
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

DEV = "cuda"; CAP = 150
torch.manual_seed(0); np.random.seed(0); random.seed(0)
rows = list(csv.DictReader(open("index.csv")))
byp = defaultdict(list)
for d in rows: byp[int(d["person"])].append(d)
sel = []
for p, lst in byp.items():
    random.shuffle(lst); sel += lst[:CAP]
random.shuffle(sel)
persons = sorted(byp.keys()); random.shuffle(persons)
test_p = set(persons[:int(len(persons)*0.3)])
def unit(v): v = np.asarray(v, np.float32); n = np.linalg.norm(v); return v/n if n > 0 else v
Y = np.array([unit([float(d["gx"]), float(d["gy"]), float(d["gz"])]) for d in sel], np.float32)
P = np.array([int(d["person"]) for d in sel])
tr = np.array([p not in test_p for p in P]); te = ~tr
print(f"이미지 {len(sel)}, person {len(persons)} (test {len(test_p)}), 개선 레시피(L2정규화+코사인)")

@torch.no_grad()
def extract(mid):
    proc = AutoImageProcessor.from_pretrained(mid); m = AutoModel.from_pretrained(mid).to(DEV).eval()
    F = []; B = 64
    for i in range(0, len(sel), B):
        imgs = [Image.open(d["path"]).convert("RGB") for d in sel[i:i+B]]
        di = proc(images=imgs, return_tensors="pt").to(DEV)
        f = torch.nn.functional.normalize(m(**di).pooler_output, dim=1)   # L2 정규화 (안정)
        F.append(f.cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(F)

def train_eval(X, name):
    Xtr = torch.tensor(X[tr]).to(DEV); Ytr = torch.tensor(Y[tr]).to(DEV)
    Xte = torch.tensor(X[te]).to(DEV); Yte = torch.tensor(Y[te]).to(DEV)
    D = X.shape[1]
    head = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 3)).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300)
    for ep in range(300):
        head.train(); opt.zero_grad()
        pr = head(Xtr); pr = pr / pr.norm(dim=1, keepdim=True)
        (((pr - Ytr)**2).sum(1).mean()).backward(); opt.step(); sch.step()
    head.eval()
    with torch.no_grad():
        pr = head(Xte); pr = pr / pr.norm(dim=1, keepdim=True)
        ang = torch.acos((pr*Yte).sum(1).clamp(-1, 1)) * 180/math.pi
    print(f"[{name:<24}] cross-person {ang.mean():.2f}° (median {ang.median():.2f}°, D={D})")
    return float(ang.mean())

res = {}
for mid in ["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large"]:
    print(f"\n--- {mid} 특징추출 ---")
    res[mid] = train_eval(extract(mid), mid)
best = min(res, key=res.get)
print(f"\n>>> 최선: {best}  ({res[best]:.2f}°)")
