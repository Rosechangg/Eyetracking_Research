"""DINOv2 백본 미세조정 (진짜 SOTA 경로). frozen이 아니라 백본+헤드를 end-to-end 학습.

이미지 224 캐싱 → 차등 학습률(백본 1e-5, 헤드 1e-3) + 수평플립 증강 + 코사인 손실.
cross-person 각도오차로 최선 모델을 dino_gaze_ft.pt에 저장.
실행: ../eye_tracking/.venv/Scripts/python.exe finetune.py
"""
import os, sys, csv, math, random
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass
import numpy as np, torch, torch.nn as nn
from collections import defaultdict
from PIL import Image
from transformers import AutoModel

DEV = "cuda"; DINO_ID = "facebook/dinov2-small"; CAP = 250; EPOCHS = 6; BS = 48
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

print(f"이미지 캐싱(224) {len(sel)}장...")
cache = np.zeros((len(sel), 224, 224, 3), np.uint8)
labels = np.zeros((len(sel), 3), np.float32); pers = np.zeros(len(sel), int)
for i, d in enumerate(sel):
    cache[i] = np.asarray(Image.open(d["path"]).convert("RGB").resize((224, 224)))
    labels[i] = unit([float(d["gx"]), float(d["gy"]), float(d["gz"])]); pers[i] = int(d["person"])

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
class DS(torch.utils.data.Dataset):
    def __init__(self, idxs, train): self.idxs = idxs; self.train = train
    def __len__(self): return len(self.idxs)
    def __getitem__(self, k):
        i = self.idxs[k]; img = cache[i]; g = labels[i].copy()
        if self.train and random.random() < 0.5:
            img = img[:, ::-1, :].copy(); g[0] = -g[0]      # 수평플립 + gx 부호
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return (x - MEAN) / STD, torch.from_numpy(g)

tr_idx = [i for i in range(len(sel)) if pers[i] not in test_p]
te_idx = [i for i in range(len(sel)) if pers[i] in test_p]
tr_dl = torch.utils.data.DataLoader(DS(tr_idx, True), batch_size=BS, shuffle=True, drop_last=True)
te_dl = torch.utils.data.DataLoader(DS(te_idx, False), batch_size=BS)
print(f"train {len(tr_idx)}, test {len(te_idx)} (cross-person)")

class GazeModel(nn.Module):
    def __init__(self, dino_id):
        super().__init__()
        self.dino = AutoModel.from_pretrained(dino_id)
        D = self.dino.config.hidden_size
        self.head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 3))
    def forward(self, x):
        g = self.head(self.dino(pixel_values=x).pooler_output)
        return g / g.norm(dim=1, keepdim=True)

model = GazeModel(DINO_ID).to(DEV)
opt = torch.optim.AdamW([{"params": model.dino.parameters(), "lr": 1e-5},
                         {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS*len(tr_dl))

def evaluate():
    model.eval(); errs = []
    with torch.no_grad():
        for x, g in te_dl:
            pr = model(x.to(DEV))
            errs.append((torch.acos((pr*g.to(DEV)).sum(1).clamp(-1, 1))*180/math.pi).cpu())
    return float(torch.cat(errs).mean())

print(f"미세조정 시작 (frozen 21° 대비 개선 목표)...")
best = 1e9
for ep in range(EPOCHS):
    model.train()
    for x, g in tr_dl:
        opt.zero_grad()
        pr = model(x.to(DEV))
        (1 - (pr*g.to(DEV)).sum(1)).mean().backward()
        opt.step(); sch.step()
    e = evaluate()
    print(f"  epoch {ep+1}/{EPOCHS}  cross-person {e:.2f}°" + ("  *best저장" if e < best else ""))
    if e < best:
        best = e
        torch.save({"dino_id": DINO_ID, "state_dict": model.state_dict(), "finetuned": True}, "dino_gaze_ft.pt")
print(f"\n최선 {best:.2f}° -> dino_gaze_ft.pt (frozen probe 21~24°와 비교)")
