"""DINOv2 라이브 시선 추적 (ZED 카메라). 3DGazeNet의 run_zed_gaze와 동일한 경험.

ZED 프레임 → insightface 얼굴검출 → 얼굴 크롭 → DINOv2 → 저장된 헤드 → 3D gaze → 화살표.
종료: 'q' 또는 ESC.

실행: ../eye_tracking/.venv/Scripts/python.exe dino_gaze_live.py
"""
import os, sys, math, ctypes
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

DEV = "cuda"
WIN = "DINOv2 gaze (q=quit)"
_U32 = ctypes.windll.user32 if sys.platform.startswith("win") else None
def key_down(vk):
    return bool(_U32.GetAsyncKeyState(vk) & 0x0001) if _U32 else False

# ---------- 모델 로드 ----------
print("헤드/DINOv2 로딩...")
ck = torch.load(os.path.join(HERE, "dino_gaze_head.pt"), map_location=DEV)
D = ck["feat_dim"]
head = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
head.load_state_dict(ck["head"]); head.to(DEV).eval()
mu, sd = ck["mu"].to(DEV), ck["sd"].to(DEV)
dino_proc = AutoImageProcessor.from_pretrained(ck["dino_id"])
dino = AutoModel.from_pretrained(ck["dino_id"]).to(DEV).eval()

print("insightface 얼굴검출 로딩...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))


@torch.no_grad()
def predict_gaze(face_bgr):
    pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    di = dino_proc(images=[pil], return_tensors="pt").to(DEV)
    f = (dino(**di).pooler_output - mu) / sd
    g = head(f); g = g / g.norm(dim=1, keepdim=True)
    return g[0].cpu().numpy()


def main():
    cam = ZEDCamera()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print("실행 중... 'q'/ESC 종료. (Gaze360 학습이라 ZED와 도메인 차이로 거칠 수 있음)")
    fps, tprev = 0.0, None
    import time
    while True:
        frame = cam.read()
        if frame is None:
            print("프레임 실패"); break
        faces = app.get(frame)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            x1, y1, x2, y2 = f.bbox.astype(int)
            mw, mh = int((x2-x1)*0.35), int((y2-y1)*0.35)        # head crop 근사 (여유 확장)
            cx1, cy1 = max(0, x1-mw), max(0, y1-mh)
            cx2, cy2 = min(frame.shape[1], x2+mw), min(frame.shape[0], y2+mh)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size:
                g = predict_gaze(crop)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                L = (x2-x1)
                dx, dy = int(g[0]*L), int(-g[1]*L)               # 화살표(부호는 필요시 조정)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.arrowedLine(frame, (cx, cy), (cx+dx, cy+dy), (0, 0, 255), 3, tipLength=0.3)
                cv2.putText(frame, f"gaze=[{g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f}]", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        now = time.time()
        if tprev:
            dt = now - tprev; fps = 0.9*fps + 0.1*(1/dt) if fps else 1/dt
        tprev = now
        cv2.putText(frame, f"FPS {fps:4.1f}  DINOv2", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WIN, frame)
        cv2.waitKey(1)
        if key_down(0x1B) or key_down(ord('Q')):
            break
    cam.release(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
