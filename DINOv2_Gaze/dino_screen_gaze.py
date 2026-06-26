"""DINOv2 화면 응시점(9점 보정 → 실시간 점). 3DGazeNet screen_gaze와 동일한 경험.

보정이 "DINOv2 출력 → 내 모니터 좌표" 매핑을 학습하므로, Gaze360 학습 헤드의
도메인 치우침/좌표계 차이도 흡수된다. 보정 후 빨간 점이 내 응시점을 실시간 추종.

조작: 보정점 보며 SPACE(전역키), r=재보정, q/ESC=종료.
실행: ../eye_tracking/.venv/Scripts/python.exe dino_screen_gaze.py
"""
import os, sys, math, ctypes, time
os.environ.setdefault("MPLBACKEND", "Agg")
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass
import cv2, numpy as np, torch, torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from insightface.app import FaceAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # zed_camera.py 동봉
from zed_camera import ZEDCamera

DEV = "cuda"; WIN = "DINOv2 screen gaze"
_U32 = ctypes.windll.user32 if sys.platform.startswith("win") else None
VK_SPACE, VK_ESC, VK_R = 0x20, 0x1B, 0x52
def key_down(vk): return bool(_U32.GetAsyncKeyState(vk) & 0x0001) if _U32 else False

def screen_size():
    try:
        ctypes.windll.user32.SetProcessDPIAware(); u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080

# ---------- 모델 (fine-tuned 우선) ----------
import torchvision.transforms as T
print("모델 로딩...")

class GazeModel(nn.Module):
    def __init__(self, dino_id):
        super().__init__()
        self.dino = AutoModel.from_pretrained(dino_id)
        Dh = self.dino.config.hidden_size
        self.head = nn.Sequential(nn.Linear(Dh, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 3))
    def forward(self, x):
        g = self.head(self.dino(pixel_values=x).pooler_output)
        return g / g.norm(dim=1, keepdim=True)

_fttf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
USE_FT = os.path.exists(os.path.join(HERE, "dino_gaze_ft.pt"))
if USE_FT:
    fk = torch.load(os.path.join(HERE, "dino_gaze_ft.pt"), map_location=DEV)
    ftmodel = GazeModel(fk["dino_id"]); ftmodel.load_state_dict(fk["state_dict"]); ftmodel.to(DEV).eval()
    print(f"[모델] fine-tuned ({fk['dino_id']}) 사용")
else:
    ck = torch.load(os.path.join(HERE, "dino_gaze_head.pt"), map_location=DEV)
    D = ck["feat_dim"]
    head = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
    head.load_state_dict(ck["head"]); head.to(DEV).eval()
    mu, sd = ck["mu"].to(DEV), ck["sd"].to(DEV)
    dino_proc = AutoImageProcessor.from_pretrained(ck["dino_id"])
    dino = AutoModel.from_pretrained(ck["dino_id"]).to(DEV).eval()
    print(f"[모델] frozen probe ({ck['dino_id']}) 사용")

@torch.no_grad()
def gaze_vec(crop_bgr):
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    if USE_FT:
        return ftmodel(_fttf(pil).unsqueeze(0).to(DEV))[0].cpu().numpy()
    di = dino_proc(images=[pil], return_tensors="pt").to(DEV)
    g = head((dino(**di).pooler_output - mu) / sd)
    return (g / g.norm(dim=1, keepdim=True))[0].cpu().numpy()
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

@torch.no_grad()
def detect_gaze(frame):
    """반환: (yaw,pitch) 또는 None, 미리보기 프레임, face_ok."""
    faces = app.get(frame)
    if not faces:
        return None, frame, False
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    x1, y1, x2, y2 = f.bbox.astype(int)
    mw, mh = int((x2-x1)*0.35), int((y2-y1)*0.35)
    cx1, cy1 = max(0, x1-mw), max(0, y1-mh)
    cx2, cy2 = min(frame.shape[1], x2+mw), min(frame.shape[0], y2+mh)
    crop = frame[cy1:cy2, cx1:cx2]
    if not crop.size:
        return None, frame, False
    g = gaze_vec(crop)
    gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
    yaw = math.degrees(math.atan2(gx, -gz)); pitch = math.degrees(math.atan2(gy, math.hypot(gx, gz)))
    prev = frame.copy()
    cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return (yaw, pitch), prev, True

def features(yaw, pitch):
    return np.array([1.0, yaw, pitch, yaw*yaw, pitch*pitch, yaw*pitch])

def thumb(canvas, prev, face_ok, W):
    tw = 360; th = int(prev.shape[0]*tw/prev.shape[1])
    col = (0, 220, 0) if face_ok else (0, 0, 255)
    x0, y0 = max(0, W-tw-20), 20
    canvas[y0:y0+th, x0:x0+tw] = cv2.resize(cv2.flip(prev, 1), (tw, th))   # 거울 표시(모델은 원본 사용)
    cv2.rectangle(canvas, (x0-2, y0-2), (x0+tw+2, y0+th+2), col, 3)
    cv2.putText(canvas, ("FACE OK" if face_ok else "NO FACE") + " (mirror)", (x0, y0+th+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

def acquire(cam, W, H, pt, label):
    key_down(VK_SPACE); buf = []
    while True:
        frame = cam.read()
        gp, prev, ok = (None, frame, False) if frame is None else detect_gaze(frame)
        c = np.zeros((H, W, 3), np.uint8)
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(c, (x, y), 22, (0, 0, 255), -1); cv2.circle(c, (x, y), 8, (255, 255, 255), -1)
        cv2.putText(c, label, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(c, "Look at dot + SPACE  (ESC=quit)", (40, H-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
        thumb(c, prev, ok, W)
        cv2.imshow(WIN, c); cv2.waitKey(1)
        if key_down(VK_ESC): return "quit"
        if gp is not None: buf.append(gp); buf = buf[-6:]
        if key_down(VK_SPACE) and buf:
            ys = [b[0] for b in buf]; ps = [b[1] for b in buf]
            return float(np.median(ys)), float(np.median(ps))

NCAL = 4          # 4x4 = 16점 보정
LAM = 1.0         # ridge 정규화 강도

def ridge(A, y, lam=LAM):
    return np.linalg.solve(A.T @ A + lam*np.eye(A.shape[1]), A.T @ y)

def map_screen(yaw, pitch, cal):
    wx, wy, fmu, fsd = cal
    f = features(yaw, pitch); f[1:] = (f[1:]-fmu)/fsd
    return float(f@wx), float(f@wy)

def calibrate(cam, W, H):
    fr = np.linspace(0.08, 0.92, NCAL)
    pts = [(fx*W, fy*H) for fy in fr for fx in fr]
    raw = []
    print(f"[보정] {NCAL*NCAL}점을 보며 SPACE")
    for i, pt in enumerate(pts):
        r = acquire(cam, W, H, pt, f"Calibration {i+1}/{NCAL*NCAL}")
        if r == "quit": return None
        raw.append((r[0], r[1], pt[0], pt[1]))
    raw = np.array(raw)
    Yaw, Pit, Xs, Ys = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    A = np.array([features(y, p) for y, p in zip(Yaw, Pit)])
    fmu, fsd = A[:, 1:].mean(0), A[:, 1:].std(0) + 1e-6      # 특징 표준화 (ridge 공정성)
    As = A.copy(); As[:, 1:] = (A[:, 1:] - fmu) / fsd
    wx, wy = ridge(As, Xs), ridge(As, Ys)
    rms = float(np.sqrt(np.mean(np.sum((np.stack([As@wx, As@wy], 1) - np.stack([Xs, Ys], 1))**2, 1))))
    print(f"[보정] Ridge(λ={LAM}) {NCAL*NCAL}점, 학습 RMS {rms:.0f}px")
    lh, th = Xs < W/2, Ys < H/2
    print(f"[진단] 좌→우 yaw 변화 {Yaw[~lh].mean()-Yaw[lh].mean():+.1f}°, "
          f"상→하 pitch 변화 {Pit[~th].mean()-Pit[th].mean():+.1f}° (|변화|<3°면 신호 약함)")
    return (wx, wy, fmu, fsd)

def main():
    W, H = screen_size(); print(f"화면 {W}x{H}")
    cam = ZEDCamera()
    cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cal = calibrate(cam, W, H)
    if cal is None: cam.release(); cv2.destroyAllWindows(); return 1
    print("실시간 응시점. r=재보정, q/ESC=종료.")
    sx, sy = W/2, H/2
    while True:
        frame = cam.read()
        gp, prev, ok = (None, frame, False) if frame is None else detect_gaze(frame)
        c = np.zeros((H, W, 3), np.uint8)
        if gp is not None:
            px, py = map_screen(gp[0], gp[1], cal)
            sx = 0.82*sx + 0.18*float(np.clip(px, 0, W-1)); sy = 0.82*sy + 0.18*float(np.clip(py, 0, H-1))
        cv2.circle(c, (int(sx), int(sy)), 26, (0, 200, 255), -1)
        cv2.circle(c, (int(sx), int(sy)), 10, (255, 255, 255), -1)
        cv2.putText(c, "LIVE: 보는 곳에 점. r=재보정 q=종료", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
        thumb(c, prev, ok, W)
        cv2.imshow(WIN, c); cv2.waitKey(1)
        if key_down(VK_ESC) or key_down(ord('Q')): break
        if key_down(VK_R):
            cal = calibrate(cam, W, H)
            if cal is None: break
    cam.release(); cv2.destroyAllWindows(); return 0

if __name__ == "__main__":
    raise SystemExit(main())
