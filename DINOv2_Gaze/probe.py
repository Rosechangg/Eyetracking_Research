"""Foundation 백본 효과 프로브: frozen DINOv2 vs frozen ResNet18 + 작은 MLP 헤드.

Gaze360(실제 다인물) 부분집합에서 cross-person(학습에 없던 사람) 각도오차를 비교한다.
백본은 둘 다 freeze, 동일한 MLP 헤드만 학습 → "백본 특징이 gaze 신호를 얼마나 담는가"를 격리 비교.

실행: ../eye_tracking/.venv/Scripts/python.exe probe.py
"""
import os, sys, csv, math, random
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass
import numpy as np, torch, torch.nn as nn
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torchvision.models as tvm
from transformers import AutoModel, AutoImageProcessor

DEV = "cuda"
CAP = 120                      # person당 최대 이미지 (속도/균형)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

# ---------- 데이터 ----------
rows = list(csv.DictReader(open("index.csv")))
byp = defaultdict(list)
for d in rows: byp[int(d["person"])].append(d)
sel = []
for p, lst in byp.items():
    random.shuffle(lst); sel += lst[:CAP]
random.shuffle(sel)
persons = sorted(byp.keys()); random.shuffle(persons)
ntest = max(1, int(len(persons) * 0.3))
test_p = set(persons[:ntest])
print(f"이미지 {len(sel)}장, person {len(persons)}명 (train {len(persons)-ntest} / test {ntest}, cross-person)")

def unit(v):
    v = np.asarray(v, np.float32); n = np.linalg.norm(v)
    return v / n if n > 0 else v

# ---------- 백본 (둘 다 frozen) ----------
print("백본 로딩 (DINOv2-small, ResNet18)...")
dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1); resnet.fc = nn.Identity()
resnet = resnet.to(DEV).eval()
res_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def extract():
    Dino, Res, Y, P = [], [], [], []
    B = 64
    for i in range(0, len(sel), B):
        chunk = sel[i:i+B]
        imgs = [Image.open(d["path"]).convert("RGB") for d in chunk]
        di = dino_proc(images=imgs, return_tensors="pt").to(DEV)
        Dino.append(dino(**di).pooler_output.cpu().numpy())
        rt = torch.stack([res_tf(im) for im in imgs]).to(DEV)
        Res.append(resnet(rt).cpu().numpy())
        for d in chunk:
            Y.append(unit([float(d["gx"]), float(d["gy"]), float(d["gz"])])); P.append(int(d["person"]))
        if i % (B*30) == 0: print(f"  특징 {i}/{len(sel)}")
    return np.concatenate(Dino), np.concatenate(Res), np.array(Y, np.float32), np.array(P)

print("특징 추출 중...")
Dino, Res, Y, P = extract()
print(f"DINOv2 특징 {Dino.shape}, ResNet 특징 {Res.shape}")
tr = np.array([p not in test_p for p in P]); te = ~tr

# ---------- 헤드 학습 + 평가 ----------
def train_eval(X, name):
    Xtr = torch.tensor(X[tr]).to(DEV); Ytr = torch.tensor(Y[tr]).to(DEV)
    Xte = torch.tensor(X[te]).to(DEV); Yte = torch.tensor(Y[te]).to(DEV)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr-mu)/sd, (Xte-mu)/sd
    D = X.shape[1]
    head = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3)).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(150):
        head.train(); opt.zero_grad()
        pr = head(Xtr); pr = pr / pr.norm(dim=1, keepdim=True)
        loss = ((pr - Ytr)**2).sum(1).mean()
        loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pr = head(Xte); pr = pr / pr.norm(dim=1, keepdim=True)
        ang = torch.acos((pr*Yte).sum(1).clamp(-1, 1)) * 180/math.pi
    print(f"[{name:<26}] cross-person 각도오차  평균 {ang.mean():.2f}°  중앙값 {ang.median():.2f}°  (특징차원 {D})")
    return float(ang.mean())

print("\n=== 헤드 학습 + 평가 (처음 보는 사람 대상) ===")
a_dino = train_eval(Dino, "DINOv2-small (frozen)")
a_res = train_eval(Res, "ResNet18-ImageNet (frozen)")
win = "DINOv2 우위" if a_dino < a_res else "ResNet 우위"
print(f"\n결론: DINOv2 {a_dino:.2f}°  vs  ResNet18 {a_res:.2f}°  ->  {win} ({abs(a_dino-a_res):.2f}° 차이)")
print("(같은 데이터·같은 헤드, 백본만 교체. 낮을수록 좋음)")
