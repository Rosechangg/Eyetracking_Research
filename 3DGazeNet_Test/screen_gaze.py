"""모니터 캘리브레이션 + 화면 응시점(Point-of-Regard) 실시간 시각화.

3DGazeNet은 3D 시선 '방향'만 주므로(메트릭 depth 없음), 화면 어디를 보는지 알려면
9점 캘리브레이션 회귀가 필요하다:
  1) 전체화면에 타겟을 차례로 띄우고, 사용자가 각 점을 보며 SPACE -> 그때의 시선
     yaw/pitch (+ 머리 위치 특징)를 수집.
  2) (yaw, pitch, 머리특징) -> (screen_x, screen_y) 릿지 정규화 2차 다항 회귀 적합.
  3) 라이브로 시선을 화면좌표로 예측해 응시점을 그려 검증.

지터/부정확 개선 포인트(이전 버전 대비):
  - 라이브에서도 단일 프레임이 아니라 짧은 링버퍼(최근 N개) 위에서 원시 gaze 벡터를
    시간 중앙값 + 이상치 제거(각도 8도 초과 컷)한 뒤 정규화하여 arctan2에 넣는다.
  - 픽셀 좌표가 아니라 (yaw, pitch) 각도 공간에서 1차 평활(EMA)한다.
  - 최종 화면점은 One-Euro 필터로 안정화(천천히 보면 강하게, 빠르게 움직이면 약하게).
  - 머리 이동 보정: centers_iris(좌/우 홍채 2D 중심)에서 눈중심과 양안거리(거리 프록시)를
    뽑아, 보정시 기준 대비 (정규화된 눈중심 이동) 2개를 회귀 특징으로 추가한다.
    centers_iris가 None이면 verts_eyes -> 검출 keypoint 순으로 안전 폴백한다.

얼굴 손실(no-face) 처리:
  - 얼굴이 안 잡히면 링버퍼는 시간 경과로 자연히 비워지고(타임스탬프 기반 만료),
    연속 미검출이 누적되면 stale로 간주하여 '얼굴 검출 안됨'을 표시한다(멈춘 커서 방지).
  - 얼굴 재획득 시 One-Euro 필터를 리셋해 공백 구간을 가로질러 보간하지 않는다.

머리를 (대략) 고정한 상태가 가장 정확하다. 머리 이동 보정을 넣었지만 큰 자세 변화는
여전히 한계가 있으니, 정확도가 나빠지면 'r'로 재보정.

사용 (기본 = 실행할 때마다 항상 처음부터 새로 보정. --load 줄 때만 저장 보정 재사용):
    .venv/Scripts/python.exe screen_gaze.py                 # 항상 새 보정 후 라이브 검증(권장)
    .venv/Scripts/python.exe screen_gaze.py --load           # (선택)저장 보정 재사용 — 머리 위치 같을 때만
    .venv/Scripts/python.exe screen_gaze.py --points 3       # 3x3=9 (기본). 5면 5x5=25
    .venv/Scripts/python.exe screen_gaze.py --log gaze_screen.tsv
키: SPACE=타겟 캡처 / r=재보정 / g=타겟토글 / s=건너뛰기 / q·ESC=종료
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import json
import time
import argparse
from collections import deque
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "3DGazeNet", "demo")
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from inference import GazeNetInference            # noqa: E402
from zed_camera import ZEDCamera, ZED_STEREO_MODES  # noqa: E402

WINDOW = "Screen Gaze Calibration (SPACE capture / r recalib / g targets / q quit)"
CALIB_PATH = os.path.join(HERE, "recordings", "calibration.json")

CALIB_VERSION = 2

# 링버퍼 표본의 최대 보존 시간(초). 이보다 오래된 gaze 표본은 stale로 폐기.
SAMPLE_MAX_AGE = 0.30
# 연속 미검출이 이 횟수를 넘으면 곧바로 'no face'로 간주.
MISS_LIMIT = 3


# ----------------- 화면 크기 -----------------
def get_screen_size():
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


# ----------------- One-Euro 필터 -----------------
class OneEuroFilter:
    """One-Euro 필터(Casiez 외, 2012). 느린 움직임에선 강하게(저지터),
    빠른 움직임에선 약하게(저지연) 평활한다. min_cutoff(상향)면 덜 부드럽고 반응 빠름,
    beta(상향)면 빠른 움직임에서 더 따라간다. 스칼라 신호 1개를 평활한다."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x, t):
        x = float(x)
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-3
        self._t_prev = t
        # 미분 추정 + 미분 평활
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat
        # 적응형 컷오프
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


# ----------------- 녹화 / PiP -----------------
class _Recorder:
    """캔버스 프레임을 .mp4로 저장. 처음 몇 프레임으로 실제 FPS를 추정해 재생속도를 맞춘다."""

    def __init__(self, path, fps=None):
        self.path = path
        self.fps = fps
        self.writer = None
        self._buf = []
        self._t0 = None

    def add(self, frame):
        if self.writer is not None:
            self.writer.write(frame)
            return
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        self._buf.append(frame)
        if self.fps is not None:
            self._open(self.fps)
        elif len(self._buf) >= 15 and (now - self._t0) > 0.5:
            self._open(len(self._buf) / (now - self._t0))

    def _open(self, fps):
        h, w = self._buf[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, max(1.0, float(fps)), (w, h))
        for f in self._buf:
            self.writer.write(f)
        self._buf = []
        print(f"[rec] 녹화 시작: {self.path} (~{fps:.1f} fps, {w}x{h})")

    def close(self):
        if self.writer is None and self._buf:
            fps = self.fps or (len(self._buf) / max(0.5, time.time() - (self._t0 or time.time())))
            self._open(fps)
        if self.writer is not None:
            self.writer.release()
            print(f"[rec] 녹화 저장 완료: {self.path}")


def draw_pip(canvas, frame, pip_width=480, margin=40, mirror=True):
    """카메라 프레임(내 영상)을 캔버스 하단 중앙에 작게 합성(PiP).
    mirror=True 면 거울처럼 좌우를 뒤집어 자연스럽게 보여준다(표시 전용 복사본만 뒤집음.
    추론/캘리브레이션에 쓰는 원본 frame 은 절대 건드리지 않는다)."""
    if frame is None:
        return
    H, W = canvas.shape[:2]
    fh, fw = frame.shape[:2]
    if fw == 0 or fh == 0:
        return
    pw = int(min(pip_width, W - 2 * margin))
    ph = int(pw * fh / fw)
    if ph <= 0 or ph > H - 2 * margin:
        return
    small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
    if mirror:
        small = cv2.flip(small, 1)   # 좌우 반전(거울). 복사본만 적용.
    x0 = (W - pw) // 2
    y0 = H - ph - margin
    canvas[y0:y0 + ph, x0:x0 + pw] = small
    cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + pw + 2, y0 + ph + 2), (0, 200, 0), 2)
    cv2.putText(canvas, "ZED (me)", (x0 + 8, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# ----------------- 시선/머리 특징 -----------------
def _eye_center_and_dist(res, kpt):
    """res(out_dict)에서 눈중심(2D 픽셀)과 양안거리 프록시(픽셀)를 안전하게 추출.
    우선순위: centers_iris -> verts_eyes[iris_idxs] -> 검출 keypoint(눈 2점).
    반환: (eye_center(2,) 또는 None, iris_dist(float) 또는 None)."""
    # 1순위: centers_iris (좌/우 홍채 2D 중심)
    ci = res.get("centers_iris") if res is not None else None
    if ci is not None and ci.get("left") is not None and ci.get("right") is not None:
        l = np.asarray(ci["left"], dtype=np.float64).ravel()[:2]
        r = np.asarray(ci["right"], dtype=np.float64).ravel()[:2]
        return (l + r) / 2.0, float(np.linalg.norm(l - r))
    # 2순위: verts_eyes 의 홍채 정점 평균
    ve = res.get("verts_eyes") if res is not None else None
    idx = res.get("iris_idxs") if res is not None else None
    if ve is not None and idx is not None and ve.get("left") is not None and ve.get("right") is not None:
        try:
            l = np.asarray(ve["left"])[idx][:, :2].mean(axis=0)
            r = np.asarray(ve["right"])[idx][:, :2].mean(axis=0)
            return (l + r) / 2.0, float(np.linalg.norm(l - r))
        except Exception:
            pass
    # 3순위: 검출 5keypoint 의 눈 2점 (kpt[0]=우안 또는 좌안, kpt[1]=반대)
    if kpt is not None:
        try:
            k = np.asarray(kpt, dtype=np.float64)
            e0 = k[0][:2]
            e1 = k[1][:2]
            return (e0 + e1) / 2.0, float(np.linalg.norm(e0 - e1))
        except Exception:
            pass
    return None, None


def poly_feats(yaw, pitch, dxn=None, dyn=None):
    """(yaw, pitch) 2차 다항 + (선택)머리이동 보정 특징.
    dxn, dyn = 정규화된 눈중심 이동(양안거리로 나눈 값). None이면 0으로 둔다.
    반환 shape: (..., 8)  [1, y, p, y^2, p^2, y*p, dxn, dyn]"""
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    if dxn is None:
        dxn = np.zeros_like(yaw)
    if dyn is None:
        dyn = np.zeros_like(yaw)
    dxn = np.asarray(dxn, dtype=np.float64)
    dyn = np.asarray(dyn, dtype=np.float64)
    return np.stack([np.ones_like(yaw), yaw, pitch,
                     yaw * yaw, pitch * pitch, yaw * pitch,
                     dxn, dyn], axis=-1)


def _gaze_to_yawpitch(g):
    """단위 gaze 벡터(3,) -> (yaw, pitch). arctan2 로 gz->0 에서도 안정적."""
    g = np.asarray(g, dtype=np.float64).ravel()
    gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
    gz = gz if abs(gz) > 1e-6 else 1e-6
    yaw = np.arctan2(gx, gz)
    pitch = np.arctan2(gy, gz)
    return yaw, pitch


def _robust_mean_vector(vecs, ang_thresh_deg=8.0):
    """원시 gaze 단위벡터들의 시간 집계: 평균 방향에서 각도 임계 초과분을 이상치로 버리고
    남은 것들을 평균 후 L2 정규화. vecs: list of (3,). 반환 (3,) 또는 None."""
    if not vecs:
        return None
    arr = np.asarray(vecs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    # 정규화
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1e-9
    arr = arr / norms
    mean = arr.mean(axis=0)
    mn = np.linalg.norm(mean)
    if mn < 1e-9:
        return None
    mean = mean / mn
    # 평균 방향과의 각도(코사인)
    cos = np.clip(arr @ mean, -1.0, 1.0)
    keep = cos >= np.cos(np.deg2rad(ang_thresh_deg))
    if keep.sum() >= 1:
        sel = arr[keep]
    else:
        sel = arr
    out = sel.mean(axis=0)
    on = np.linalg.norm(out)
    if on < 1e-9:
        return None
    return out / on


class GazeReader:
    """가장 큰 얼굴 하나의 원시 gaze 벡터 + 머리 특징을 반환.
    링버퍼 시간 집계는 호출자(stream)에서 수행한다."""

    def __init__(self, gazenet):
        self.g = gazenet

    def read_raw(self, frame):
        """반환: dict(gaze_vec(3,), eye_center(2,)|None, iris_dist(float)|None) 또는 None."""
        if frame is None:
            return None
        try:
            bboxs, kpts, _ = self.g.face_detector.run(frame)
        except Exception:
            return None
        if kpts is None or len(kpts) == 0:
            return None
        # 가장 큰 얼굴 하나만
        try:
            areas = [float((b[2] - b[0]) * (b[3] - b[1])) for b in bboxs]
            i = int(np.argmax(areas))
        except Exception:
            i = 0
        kpt = kpts[i]
        try:
            res = self.g.gaze_predictor(frame, kpt, undo_roll=True)
        except Exception:
            return None
        if res is None:
            return None
        # 신호 선택: gaze_out(=vertex 모드에서 gaze_combined) 우선, 폴백 체인
        g = res.get("gaze_out")
        if g is None:
            g = res.get("gaze_combined")
        if g is None:
            gfe = res.get("gaze_from_eyes")
            if gfe is not None:
                g = gfe.get("face")
        if g is None:
            g = res.get("gaze")
        if g is None:
            return None
        gaze_vec = np.asarray(g, dtype=np.float64).ravel()
        if gaze_vec.shape[0] < 3:
            return None
        ec, idist = _eye_center_and_dist(res, kpt)
        return {"gaze_vec": gaze_vec[:3], "eye_center": ec, "iris_dist": idist}


class GazeStream:
    """프레임 -> 시간 집계된 (yaw, pitch, eye_center, iris_dist) 를 내는 스트림.
    최근 buf_len개의 원시 gaze 벡터를 모아 robust mean(이상치 제거) 후 arctan2.
    각도 공간에서 EMA(alpha)로 1차 평활. eye_center/iris_dist 도 짧은 중앙값으로 안정화.

    링버퍼 항목은 (timestamp, value) 쌍으로 저장한다. 매 push 마다 max_age 보다
    오래된 항목을 만료시키므로, 얼굴이 사라지면 버퍼가 자연히 비워져 stale 값이
    무한히 남지 않는다. 또한 연속 미검출(miss) 횟수를 추적한다."""

    def __init__(self, reader, buf_len=6, ang_thresh_deg=8.0, angle_ema=0.35,
                 max_age=SAMPLE_MAX_AGE, miss_limit=MISS_LIMIT):
        self.reader = reader
        self.buf_len = int(buf_len)
        self.ang_thresh_deg = float(ang_thresh_deg)
        self.angle_ema = float(angle_ema)
        self.max_age = float(max_age)
        self.miss_limit = int(miss_limit)
        self._gbuf = deque(maxlen=self.buf_len)   # (t, gaze_vec)
        self._ecbuf = deque(maxlen=self.buf_len)  # (t, eye_center)
        self._dbuf = deque(maxlen=self.buf_len)   # (t, iris_dist)
        self._yaw_s = None
        self._pitch_s = None
        self.miss_count = 0

    def reset(self):
        self._gbuf.clear()
        self._ecbuf.clear()
        self._dbuf.clear()
        self._yaw_s = None
        self._pitch_s = None
        self.miss_count = 0

    def _expire(self, now):
        """now 기준으로 max_age 보다 오래된 항목을 앞쪽부터 제거(타임스탬프 만료)."""
        cut = now - self.max_age
        for buf in (self._gbuf, self._ecbuf, self._dbuf):
            while buf and buf[0][0] < cut:
                buf.popleft()

    def push(self, frame):
        """프레임 1장 처리하여 버퍼에 적재. 반환: True(얼굴 잡힘)/False(미검출).
        미검출이면 버퍼에 아무것도 넣지 않고 miss_count 를 증가시킨다.
        성공/실패 무관하게 오래된 표본은 만료시킨다(stale 방지)."""
        now = time.time()
        raw = self.reader.read_raw(frame)
        if raw is None:
            self.miss_count += 1
            # 얼굴이 없으면 새로 넣지 않되, 기존 표본도 시간 경과로 만료시킨다.
            self._expire(now)
            return False
        self.miss_count = 0
        self._gbuf.append((now, np.asarray(raw["gaze_vec"], dtype=np.float64).ravel()[:3]))
        if raw["eye_center"] is not None:
            self._ecbuf.append((now, np.asarray(raw["eye_center"], dtype=np.float64).ravel()[:2]))
        if raw["iris_dist"] is not None and np.isfinite(raw["iris_dist"]):
            self._dbuf.append((now, float(raw["iris_dist"])))
        self._expire(now)
        return True

    def is_stale(self):
        """현재 값이 신뢰할 수 없는 상태인지. 연속 미검출이 한계를 넘었거나
        시간 만료로 유효 표본이 0개면 stale."""
        if self.miss_count >= self.miss_limit:
            return True
        if len(self._gbuf) == 0:
            return True
        return False

    def value(self):
        """현재 집계 결과. stale(미검출 누적/표본 만료)이면 None.
        반환: dict 또는 None.
        keys: yaw, pitch (각 EMA 적용), eye_center(2,)|None, iris_dist(float)|None."""
        if self.is_stale():
            return None
        vecs = [v for (_, v) in self._gbuf]
        mv = _robust_mean_vector(vecs, self.ang_thresh_deg)
        if mv is None:
            return None
        yaw, pitch = _gaze_to_yawpitch(mv)
        # 각도 공간 EMA
        a = self.angle_ema
        if self._yaw_s is None:
            self._yaw_s, self._pitch_s = yaw, pitch
        else:
            self._yaw_s = a * yaw + (1 - a) * self._yaw_s
            self._pitch_s = a * pitch + (1 - a) * self._pitch_s
        ec = None
        if len(self._ecbuf) > 0:
            ec = np.median(np.asarray([v for (_, v) in self._ecbuf]), axis=0)
        idist = None
        if len(self._dbuf) > 0:
            idist = float(np.median(np.asarray([v for (_, v) in self._dbuf])))
        return {"yaw": float(self._yaw_s), "pitch": float(self._pitch_s),
                "eye_center": ec, "iris_dist": idist}


def target_grid(W, H, n):
    """n x n 타겟 좌표 (가장자리 8% 여백). 가운데->바깥 순으로 정렬해 시작이 편하게."""
    fr = np.linspace(0.08, 0.92, n)
    pts = [(int(fx * W), int(fy * H)) for fy in fr for fx in fr]
    c = (W / 2, H / 2)
    pts.sort(key=lambda p: (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)
    return pts


def draw_target(canvas, pt, phase):
    x, y = pt
    r = int(18 + 6 * np.sin(phase))          # 맥동
    cv2.circle(canvas, (x, y), r + 8, (40, 40, 40), 2)
    cv2.circle(canvas, (x, y), r, (0, 0, 255), -1)
    cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)


# ----------------- dwell 기반 캡처 -----------------
def dwell_capture(cam, stream, target, W, H,
                  n_collect=28, n_min=12, timeout=6.0):
    """SPACE 이후 호출. 짧은 dwell 동안 원시 표본을 모으고 MAD 이상치/깜빡임 제거 후
    yaw/pitch 중앙값 + 머리 특징 중앙값을 반환. 표본이 n_min 미만이면 None(재시도).

    dwell 동안에도 매 반복 imshow + waitKey(1)로 창 이벤트 루프를 펌프해
    'Not Responding' 을 방지하고, q/ESC 로 즉시 중단/조기 종료할 수 있게 한다.
    반환: (out_dict 또는 None, aborted(bool))."""
    ys, ps, ecs, dists = [], [], [], []
    t0 = time.time()
    aborted = False
    # 캡처 동안엔 EMA 영향을 배제하려고 신선한 원시값을 직접 모은다.
    while len(ys) < n_collect and (time.time() - t0) < timeout:
        frame = cam.read()
        raw = stream.reader.read_raw(frame)
        face_ok = raw is not None
        if face_ok:
            yaw, pitch = _gaze_to_yawpitch(raw["gaze_vec"])
            ys.append(yaw)
            ps.append(pitch)
            if raw["eye_center"] is not None:
                ecs.append(np.asarray(raw["eye_center"], dtype=np.float64).ravel()[:2])
            if raw["iris_dist"] is not None and np.isfinite(raw["iris_dist"]):
                dists.append(float(raw["iris_dist"]))

        # 창을 계속 갱신해 응답성 유지 + 라이브 피드백 + 중단 폴링
        canvas = np.zeros((H, W, 3), np.uint8)
        draw_target(canvas, target, time.time() * 4)
        elapsed = time.time() - t0
        cv2.putText(canvas,
                    f"캡처 중... 계속 응시 ({len(ys)}/{n_collect}, {elapsed:0.1f}s)",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, "face: " + ("OK" if face_ok else "검출안됨"),
                    (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if face_ok else (0, 0, 255), 2)
        cv2.putText(canvas, "q/ESC=중단", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.imshow(WINDOW, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            aborted = True
            break

    if aborted:
        return None, True
    if len(ys) < n_min:
        return None, False

    ys = np.asarray(ys, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)

    def _mad_keep(v, k=3.0):
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad < 1e-9:
            return np.ones_like(v, dtype=bool)
        return np.abs(v - med) <= k * 1.4826 * mad

    keep = _mad_keep(ys) & _mad_keep(ps)
    if keep.sum() < n_min:
        # 너무 많이 버려지면(깜빡임/흔들림) 그래도 남은 전체로 진행하되 실패는 아니게
        keep = np.ones_like(ys, dtype=bool)
    ys2, ps2 = ys[keep], ps[keep]

    out = {
        "yaw": float(np.median(ys2)),
        "pitch": float(np.median(ps2)),
        "n": int(keep.sum()),
        "eye_center": (np.median(np.asarray(ecs), axis=0).tolist() if len(ecs) else None),
        "iris_dist": (float(np.median(np.asarray(dists))) if len(dists) else None),
    }
    return out, False


# ----------------- 릿지 다항 회귀 + 표준화 -----------------
def _standardize_fit(F):
    """특징 행렬 F(N,8)의 평균/표준편차. 절편(0열)은 표준화 제외."""
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    mu[0] = 0.0
    sd[0] = 1.0
    sd[sd < 1e-9] = 1.0
    return mu, sd


def _apply_std(F, mu, sd):
    return (F - mu) / sd


def ridge_fit(Fz, y, lam=1e-2):
    """표준화된 Fz(N,d)에 대해 (Fz^T Fz + lam*I) w = Fz^T y. 절편(0열)은 정규화 제외."""
    d = Fz.shape[1]
    A = Fz.T @ Fz
    reg = lam * np.eye(d)
    reg[0, 0] = 0.0
    A = A + reg
    b = Fz.T @ y
    w = np.linalg.solve(A, b)
    return w


# lambda 자동 선택용 그리드(누설 없는 LOO 최소화로 선택)
LAM_GRID = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]


def _loo_px(F, sx, sy, lam, n):
    """누설(leakage) 없는 Leave-One-Out CV 오차를 계산한다.

    각 fold 마다 학습 표본만으로 mu_i/sd_i 를 재추정해 학습/테스트 점을 표준화하고
    가중치를 적합한다 -> hold-out 표본이 표준화 통계에 새지 않아 진짜 out-of-sample.

    반환: (loo_px, per_point_resid(len n, 미계산 점은 nan), loo_x_px, loo_y_px)."""
    resid = np.full(n, np.nan, dtype=np.float64)   # 점별 LOO 잔차(px). 축별 분해용으로 dx,dy 별도 보관.
    dxs, dys = [], []
    if n >= 4:
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            F_tr = F[mask]
            mu_i, sd_i = _standardize_fit(F_tr)       # 학습 fold 로만 표준화 통계 추정
            Fz_tr = _apply_std(F_tr, mu_i, sd_i)
            Fz_te = _apply_std(F[i], mu_i, sd_i)      # 테스트 점도 동일 통계로 표준화
            try:
                wxi = ridge_fit(Fz_tr, sx[mask], lam)
                wyi = ridge_fit(Fz_tr, sy[mask], lam)
            except Exception:
                continue
            ex = float(Fz_te @ wxi) - sx[i]
            ey = float(Fz_te @ wyi) - sy[i]
            resid[i] = float(np.hypot(ex, ey))
            dxs.append(abs(ex))
            dys.append(abs(ey))
    valid = resid[np.isfinite(resid)]
    loo_px = float(np.mean(valid)) if valid.size else float("nan")
    loo_x_px = float(np.mean(dxs)) if dxs else float("nan")
    loo_y_px = float(np.mean(dys)) if dys else float("nan")
    return loo_px, resid, loo_x_px, loo_y_px


def fit_calibration(samples, W, H, lam=None):
    """samples: list of dict(yaw,pitch,sx,sy,eye_center|None,iris_dist|None).
    머리 특징(dxn,dyn) = 정규화된 눈중심 이동(기준 대비, 양안거리로 나눔)을 포함해 적합.
    LOO CV 오차(px, deg)를 함께 계산해 반환.

    lam: 숫자면 그 값 고정(back-compat). None 또는 음수면 'auto' -> LAM_GRID 위에서
    누설 없는 LOO(loo_px)를 최소화하는 lam 을 자동 선택한다.

    또한 best lam 의 점별 LOO 잔차로 이상치(MAD 3시그마) 보정점을 제거한 뒤
    남은 점으로 재적합한다(안전장치: 최소 표본/최대 제거 비율 제한)."""
    auto = (lam is None) or (isinstance(lam, (int, float)) and lam < 0)

    def _build(samples_in):
        """표본 리스트 -> (F, sx, sy, head 메타). 머리 기준/특징을 표본 내에서 추정."""
        n = len(samples_in)
        ys = np.array([s["yaw"] for s in samples_in], dtype=np.float64)
        ps = np.array([s["pitch"] for s in samples_in], dtype=np.float64)
        sx = np.array([s["sx"] for s in samples_in], dtype=np.float64)
        sy = np.array([s["sy"] for s in samples_in], dtype=np.float64)

        # 머리 기준: 눈중심/양안거리의 (있는 표본) 중앙값
        ecs = [np.asarray(s["eye_center"], dtype=np.float64) for s in samples_in if s.get("eye_center") is not None]
        dists = [float(s["iris_dist"]) for s in samples_in if s.get("iris_dist") is not None]
        have_head = len(ecs) >= max(3, n // 2) and len(dists) >= max(3, n // 2)
        if have_head:
            ec_ref = np.median(np.asarray(ecs), axis=0)
            d_ref = float(np.median(np.asarray(dists)))
            d_ref = d_ref if d_ref > 1e-6 else 1.0
        else:
            ec_ref = None
            d_ref = None

        def _head_feats(s):
            if not have_head or s.get("eye_center") is None:
                return 0.0, 0.0
            ec = np.asarray(s["eye_center"], dtype=np.float64).ravel()[:2]
            d = float(s["iris_dist"]) if s.get("iris_dist") is not None else d_ref
            d = d if d and d > 1e-6 else d_ref
            dxy = ec - ec_ref
            return float(dxy[0] / d), float(dxy[1] / d)

        dxn = np.array([_head_feats(s)[0] for s in samples_in], dtype=np.float64)
        dyn = np.array([_head_feats(s)[1] for s in samples_in], dtype=np.float64)
        F = poly_feats(ys, ps, dxn, dyn)              # (n, 8)
        return F, ys, ps, sx, sy, have_head, ec_ref, d_ref

    def _select_lam(F, sx, sy, n):
        """auto 면 LAM_GRID 위에서 loo_px 최소 lam 선택, 고정이면 그대로."""
        if not auto:
            lam_sel = float(lam)
            lp, resid, lx, ly = _loo_px(F, sx, sy, lam_sel, n)
            return lam_sel, lp, resid, lx, ly
        best = None
        for cand in LAM_GRID:
            lp, resid, lx, ly = _loo_px(F, sx, sy, cand, n)
            score = lp if np.isfinite(lp) else float("inf")
            if best is None or score < best[0]:
                best = (score, cand, lp, resid, lx, ly)
        # 표본이 너무 적어 LOO 미계산(전부 nan)이면 그리드 중간값으로 폴백.
        if best is None or not np.isfinite(best[0]):
            lam_sel = 1e-2
            lp, resid, lx, ly = _loo_px(F, sx, sy, lam_sel, n)
            return lam_sel, lp, resid, lx, ly
        return best[1], best[2], best[3], best[4], best[5]

    # 1차: 전체 표본으로 lambda 선택 + 점별 LOO 잔차 확보
    F, ys, ps, sx, sy, have_head, ec_ref, d_ref = _build(samples)
    n = len(samples)
    lam_sel, loo_px, resid, loo_x_px, loo_y_px = _select_lam(F, sx, sy, n)

    # 2차: 이상치 보정점 제거(MAD 3시그마). 안전장치로 남은 점/제거 비율 제한.
    #  제거 후에는 best lam 을 그대로 재사용하지 않고 lambda 를 다시 선택한다(점 분포 변화 반영).
    n_dropped = 0
    dropped_idx = []
    kept_samples = list(samples)
    valid = resid[np.isfinite(resid)]
    if valid.size >= 4:
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med)))
        thr = med + 3.0 * 1.4826 * mad
        if mad > 1e-9:
            bad = [i for i in range(n) if np.isfinite(resid[i]) and resid[i] > thr]
            n_keep = n - len(bad)
            min_keep = max(9, int(np.ceil(0.7 * n)))
            if bad and n_keep >= min_keep and len(bad) <= int(np.floor(0.25 * n)):
                dropped_idx = sorted(bad)
                n_dropped = len(dropped_idx)
                kept_samples = [s for i, s in enumerate(samples) if i not in set(dropped_idx)]
                # 남은 점으로 재구성 후 lambda 재선택 + LOO 재계산
                F, ys, ps, sx, sy, have_head, ec_ref, d_ref = _build(kept_samples)
                n = len(kept_samples)
                lam_sel, loo_px, resid, loo_x_px, loo_y_px = _select_lam(F, sx, sy, n)

    # 최종 모델: (남은) 전체 표본으로 표준화 + 적합
    mu, sd = _standardize_fit(F)                  # 최종 모델용(전체 표본) 표준화 통계
    Fz = _apply_std(F, mu, sd)
    wx = ridge_fit(Fz, sx, lam_sel)
    wy = ridge_fit(Fz, sy, lam_sel)

    pred = np.stack([Fz @ wx, Fz @ wy], axis=1)
    tgt = np.stack([sx, sy], axis=1)
    rms = float(np.sqrt(np.mean(np.sum((pred - tgt) ** 2, axis=1))))

    if not np.isfinite(loo_px):
        loo_px = rms

    # px -> deg 환산: 보정 표본의 yaw/pitch 각도 변화량 대비 화면 픽셀 변화량으로 스케일 추정
    span_ang = np.hypot(np.ptp(ys), np.ptp(ps))          # 라디안 대각 범위
    span_px = np.hypot(np.ptp(sx), np.ptp(sy))           # 픽셀 대각 범위
    if span_ang > 1e-6 and span_px > 1e-6:
        px_per_deg = (span_px / span_ang) * (np.pi / 180.0)
        loo_deg = float(loo_px / px_per_deg) if px_per_deg > 1e-6 else float("nan")
    else:
        loo_deg = float("nan")

    # 신호 범위 진단(deg): yaw/pitch 가 실제로 얼마나 변했는지(수직 신호 약함 탐지용)
    yaw_range_deg = float(np.ptp(ys) * 180.0 / np.pi)
    pitch_range_deg = float(np.ptp(ps) * 180.0 / np.pi)

    # 사후 진단용 원시 표본(KEPT) 저장. numpy 타입 제거하여 JSON 직렬화 보장.
    samples_json = []
    for s in kept_samples:
        ec = s.get("eye_center")
        idist = s.get("iris_dist")
        samples_json.append({
            "yaw": float(s["yaw"]), "pitch": float(s["pitch"]),
            "sx": float(s["sx"]), "sy": float(s["sy"]),
            "eye_center": ([float(v) for v in np.asarray(ec).ravel()[:2]] if ec is not None else None),
            "iris_dist": (float(idist) if idist is not None else None),
        })

    calib = {
        "version": CALIB_VERSION,
        "W": int(W), "H": int(H),
        "lam": float(lam_sel),
        "wx": wx.tolist(), "wy": wy.tolist(),
        "mu": mu.tolist(), "sd": sd.tolist(),
        "have_head": bool(have_head),
        "ec_ref": (ec_ref.tolist() if ec_ref is not None else None),
        "d_ref": (d_ref if d_ref is not None else None),
        "rms_px": rms,
        "loo_px": loo_px,
        "loo_deg": loo_deg,
        "loo_x_px": loo_x_px,
        "loo_y_px": loo_y_px,
        "yaw_range_deg": yaw_range_deg,
        "pitch_range_deg": pitch_range_deg,
        "lam_selected": float(lam_sel),
        "n_dropped": int(n_dropped),
        "dropped_idx": [int(i) for i in dropped_idx],
        "n": n,
        "samples": samples_json,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    return calib


def _head_feats_live(calib, eye_center, iris_dist):
    """라이브에서 머리 특징(dxn,dyn) 계산. 보정에 머리정보가 없거나 현재 눈중심이
    없으면 0,0 (보정 영향 없음)."""
    if not calib.get("have_head") or calib.get("ec_ref") is None or eye_center is None:
        return 0.0, 0.0
    ec = np.asarray(eye_center, dtype=np.float64).ravel()[:2]
    ec_ref = np.asarray(calib["ec_ref"], dtype=np.float64).ravel()[:2]
    d_ref = calib.get("d_ref") or 1.0
    d = iris_dist if (iris_dist is not None and iris_dist > 1e-6) else d_ref
    d = d if d and d > 1e-6 else 1.0
    dxy = ec - ec_ref
    return float(dxy[0] / d), float(dxy[1] / d)


def predict_screen(calib, yaw, pitch, eye_center=None, iris_dist=None):
    dxn, dyn = _head_feats_live(calib, eye_center, iris_dist)
    F = poly_feats(np.array([yaw]), np.array([pitch]),
                   np.array([dxn]), np.array([dyn]))[0]
    mu = np.asarray(calib["mu"], dtype=np.float64)
    sd = np.asarray(calib["sd"], dtype=np.float64)
    Fz = (F - mu) / sd
    sx = float(np.dot(Fz, calib["wx"]))
    sy = float(np.dot(Fz, calib["wy"]))
    return sx, sy


# ----------------- 캘리브레이션 루프 -----------------
def calibrate(cam, stream, W, H, n, lam,
              n_collect=28, n_min=12):
    targets = target_grid(W, H, n)
    samples = []   # dict(yaw,pitch,sx,sy,eye_center,iris_dist)
    # 플래그(WINDOW_NORMAL=0)로 창을 만든 뒤, 속성으로 전체화면을 설정한다.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    idx = 0
    print(f"[calib] {len(targets)}개 타겟. 각 점을 바라보며 SPACE. (s=건너뛰기, q=취소)")
    while idx < len(targets):
        # 라이브 미리보기: 타겟 + 얼굴 검출 상태
        frame = cam.read()
        face_ok = stream.reader.read_raw(frame) is not None
        canvas = np.zeros((H, W, 3), np.uint8)
        draw_target(canvas, targets[idx], time.time() * 4)
        msg = f"[{idx + 1}/{len(targets)}] 이 점을 응시하고 SPACE (s=skip, q=quit)"
        cv2.putText(canvas, msg, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        cv2.putText(canvas, "face: " + ("OK" if face_ok else "검출안됨"),
                    (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if face_ok else (0, 0, 255), 2)
        cv2.imshow(WINDOW, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return None
        if k == ord("s"):
            idx += 1
            continue
        if k == 32:  # SPACE
            s, aborted = dwell_capture(cam, stream, targets[idx], W, H,
                                       n_collect=n_collect, n_min=n_min)
            if aborted:
                print("[calib] 사용자 중단(q/ESC).")
                return None
            if s is None:
                print(f"  타겟 {idx + 1}: 표본 부족(<{n_min}) -> 같은 점 재시도")
                # 짧은 안내 후 동일 타겟 재시도(idx 증가 안 함)
                canvas2 = np.zeros((H, W, 3), np.uint8)
                draw_target(canvas2, targets[idx], time.time() * 4)
                cv2.putText(canvas2, "표본 부족: 다시 SPACE", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.imshow(WINDOW, canvas2)
                cv2.waitKey(400)
                continue
            sx, sy = targets[idx]
            s["sx"], s["sy"] = float(sx), float(sy)
            samples.append(s)
            print(f"  타겟 {idx + 1}/{len(targets)} 캡처 "
                  f"(n={s['n']}, yaw={s['yaw']:+.3f}, pitch={s['pitch']:+.3f}, "
                  f"head={'O' if s.get('eye_center') is not None else 'X'})")
            idx += 1

    if len(samples) < 6:
        print(f"[calib] 표본 부족({len(samples)}) -> 보정 실패")
        return None

    calib = fit_calibration(samples, W, H, lam)
    calib["targets"] = [list(t) for t in targets]
    diag = np.hypot(W, H)
    deg_str = (f"{calib['loo_deg']:.2f}deg" if np.isfinite(calib["loo_deg"]) else "deg N/A")
    n_kept = calib.get("n", len(samples))
    nd = calib.get("n_dropped", 0)
    drop_str = (f" (이상치 {nd}개 제거 -> {n_kept})" if nd else "")
    print(f"[calib] 완료. 표본 {len(samples)}{drop_str}, "
          f"head보정={'ON' if calib['have_head'] else 'OFF'}")
    print(f"[calib] lambda={calib['lam_selected']:.3g} | 학습 RMS {calib['rms_px']:.1f}px "
          f"| LOO-CV {calib['loo_px']:.1f}px ({100 * calib['loo_px'] / diag:.1f}% of diag) | {deg_str}")
    lx = calib.get("loo_x_px", float("nan"))
    ly = calib.get("loo_y_px", float("nan"))
    print(f"[calib] LOO 축별: x {lx:.1f}px / y {ly:.1f}px | "
          f"신호범위 yaw {calib['yaw_range_deg']:.1f}deg / pitch {calib['pitch_range_deg']:.1f}deg")
    # 휴리스틱 경고: 수직 신호 약함 / 정확도 불량(머리 움직임 의심)
    pr = calib.get("pitch_range_deg", 0.0)
    yr = calib.get("yaw_range_deg", 0.0)
    if pr < 4.0 or pr < 0.5 * yr:
        print("[calib] 경고: 수직(상하) 시선 신호가 약함 -> 카메라 위치/머리각도 조정 권장.")
    if np.isfinite(calib["loo_deg"]) and calib["loo_deg"] > 5.0:
        print("[calib] 경고: 정확도 불량(LOO > 5deg). 캡처 중 머리가 움직였을 가능성 -> "
              "머리를 고정하고 재보정 권장.")
    return calib


# ----------------- 라이브 검증 -----------------
def verify(cam, stream, calib, args, show_targets=True, log_path=None, recorder=None):
    W, H = calib["W"], calib["H"]
    no_display = getattr(args, "no_display", False)
    show_pip = not getattr(args, "no_pip", False)
    if not no_display:
        # 플래그(WINDOW_NORMAL=0)로 창을 만든 뒤, 속성으로 전체화면을 설정한다.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    if log:
        log.write("timestamp\tyaw\tpitch\tscreen_x\tscreen_y\n")
    # recorder 는 main 에서 생성/종료(finally)한다 -> 예외/중단에도 저장 보장.
    if recorder is not None:
        print(f"[rec] 이 라이브 세션은 녹화됩니다 -> {recorder.path}  ('q'/ESC 종료 시 저장)")
    frame_idx = 0

    # 화면점 One-Euro 필터(x,y 각각)
    fx = OneEuroFilter(min_cutoff=args.filter_min_cutoff, beta=args.filter_beta)
    fy = OneEuroFilter(min_cutoff=args.filter_min_cutoff, beta=args.filter_beta)
    stream.reset()

    targets = calib.get("targets", [])
    diag = np.hypot(W, H)
    deg_str = (f"{calib.get('loo_deg', float('nan')):.2f}deg"
               if np.isfinite(calib.get("loo_deg", float("nan"))) else "deg N/A")
    loo_px = calib.get("loo_px", calib.get("rms_px", 0.0))
    print("[verify] 화면을 둘러보며 점이 시선을 따라오는지 확인. r=재보정, g=타겟토글, q=종료")
    recalib = False
    had_face = False   # 직전 프레임에서 얼굴이 유효했는지(재획득 시 필터 리셋용)
    while True:
        frame = cam.read()
        got = stream.push(frame)
        canvas = np.zeros((H, W, 3), np.uint8)
        if show_targets:
            for (tx, ty) in targets:
                cv2.circle(canvas, (int(tx), int(ty)), 16, (60, 60, 60), 2)

        # 얼굴이 잡혔을 때만 값 사용. 미검출/만료면 value()가 None -> 'no face' 분기.
        val = stream.value() if got else None
        now = time.time()
        if val is not None:
            # 얼굴 재획득 직후엔 필터를 리셋해 공백 구간을 가로질러 보간하지 않는다.
            if not had_face:
                fx.reset()
                fy.reset()
            had_face = True

            sx, sy = predict_screen(calib, val["yaw"], val["pitch"],
                                    val["eye_center"], val["iris_dist"])
            rx = float(np.clip(sx, 0, W - 1))
            ry = float(np.clip(sy, 0, H - 1))
            # One-Euro 평활(화면점)
            px = float(np.clip(fx(rx, now), 0, W - 1))
            py = float(np.clip(fy(ry, now), 0, H - 1))

            # --- 진단 오버레이: 신호(yaw/pitch)가 실제로 반응하는지 보이게 ---
            yaw_d = float(np.degrees(val["yaw"]))
            pitch_d = float(np.degrees(val["pitch"]))
            cv2.putText(canvas,
                        f"yaw {yaw_d:+5.1f}  pitch {pitch_d:+5.1f} (deg)  raw_y={int(ry)}",
                        (40, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # 우측 수직 게이지: pitch(-25~+25도)를 화면 위->아래로 표시.
            # 위/아래를 볼 때 이 노란 점이 위아래로 움직이면 'pitch 신호 정상'(=문제는 보정),
            # 거의 안 움직이면 '모델/카메라 위치상 수직 신호 부족'.
            gx0 = W - 70
            cv2.line(canvas, (gx0, 120), (gx0, H - 140), (90, 90, 90), 2)
            pr = float(np.clip((pitch_d + 25.0) / 50.0, 0.0, 1.0))
            gyc = int(120 + pr * (H - 260))
            cv2.circle(canvas, (gx0, gyc), 13, (0, 255, 255), -1)
            cv2.putText(canvas, "pitch", (gx0 - 40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 흐린 원시점(지터 가시화)
            cv2.drawMarker(canvas, (int(rx), int(ry)), (0, 90, 90),
                           cv2.MARKER_TILTED_CROSS, 22, 1)
            # 필터된 점(초록 크로스헤어)
            cv2.circle(canvas, (int(px), int(py)), 26, (0, 255, 0), 3)
            cv2.drawMarker(canvas, (int(px), int(py)), (0, 255, 0),
                           cv2.MARKER_CROSS, 40, 2)
            if log:
                log.write(f"{now:.6f}\t{val['yaw']:.5f}\t{val['pitch']:.5f}\t"
                          f"{px:.1f}\t{py:.1f}\n")
        else:
            # 얼굴 손실/표본 만료: 멈춘 커서 대신 명확히 안내.
            # 다음 재획득에서 필터가 리셋되도록 had_face 를 내린다.
            had_face = False
            cv2.putText(canvas, "얼굴 검출 안됨", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 정확도/안내 HUD
        cv2.putText(canvas,
                    f"LOO {loo_px:.0f}px ({100 * loo_px / diag:.1f}% diag) | {deg_str} | "
                    f"head {'ON' if calib.get('have_head') else 'OFF'}",
                    (40, H - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 120), 2)
        cv2.putText(canvas,
                    "r=recalib  g=targets  q=quit  (흐린점=raw, 초록=filtered)",
                    (40, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # 내 영상 PiP (하단 중앙) -> 그 다음 녹화하면 PiP까지 영상에 들어간다
        if show_pip:
            draw_pip(canvas, frame, pip_width=getattr(args, "pip_width", 480),
                     mirror=not getattr(args, "no_mirror", False))
        if recorder is not None:
            scale = getattr(args, "record_scale", 0.5)
            rec_frame = canvas
            if scale and scale != 1.0:
                rec_frame = cv2.resize(canvas, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
            recorder.add(rec_frame)

        if not no_display:
            cv2.imshow(WINDOW, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("g"):
                show_targets = not show_targets
            if k == ord("r"):
                recalib = True
                break
        frame_idx += 1
        if getattr(args, "max_frames", 0) and frame_idx >= args.max_frames:
            break
    if log:
        log.close()
        print(f"[verify] 응시점 로그 저장: {log_path}")
    return recalib


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="모니터 캘리브레이션 + 화면 응시점 시각화")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--resolution", default="HD720", choices=list(ZED_STEREO_MODES.keys()))
    ap.add_argument("--backend", default="auto", choices=["auto", "opencv", "pyzed"])
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--eye", default="left", choices=["left", "right"])
    ap.add_argument("--points", type=int, default=3, help="격자 한 변의 점 개수 (3 -> 3x3=9)")
    ap.add_argument("--det-size", type=int, default=224)
    ap.add_argument("--load", action="store_true", help="저장된 보정 불러와 바로 검증")
    ap.add_argument("--recalibrate", action="store_true",
                    help="저장된 보정 무시하고 처음부터 새로 보정 (--load 보다 우선)")
    ap.add_argument("--log", default=None, help="응시점 좌표 로그(.tsv) 저장 경로")
    # 추가 플래그(지터 튜닝): One-Euro 파라미터 + 회귀 정규화
    ap.add_argument("--filter-min-cutoff", type=float, default=1.0,
                    help="One-Euro min_cutoff (작을수록 더 부드럽고 더 지연)")
    ap.add_argument("--filter-beta", type=float, default=0.007,
                    help="One-Euro beta (클수록 빠른 움직임을 더 따라감)")
    ap.add_argument("--ridge-lam", type=float, default=-1.0,
                    help="릿지 회귀 정규화 강도(기본 -1=auto: LOO 최소화로 자동 선택. "
                         "양수 지정 시 그 값 고정)")
    ap.add_argument("--buf-len", type=int, default=6,
                    help="원시 gaze 시간 집계 링버퍼 길이")
    ap.add_argument("--angle-ema", type=float, default=0.35,
                    help="yaw/pitch 각도 공간 EMA alpha")
    # 녹화 / PiP (내 영상)
    ap.add_argument("--session", action="store_true",
                    help="recordings/session_<시각>/ 에 screen.mp4 + gaze.tsv 자동 저장")
    ap.add_argument("--record", default=None, metavar="PATH.mp4",
                    help="화면(시각화 + 내 영상 PiP) 녹화 경로")
    ap.add_argument("--record-scale", type=float, default=0.5,
                    help="녹화 다운스케일 비율 (0.5=절반, 파일/속도 절약)")
    ap.add_argument("--pip-width", type=int, default=480, help="내 영상 PiP 가로 픽셀")
    ap.add_argument("--no-pip", action="store_true", help="내 영상 PiP 끄기")
    ap.add_argument("--no-mirror", action="store_true",
                    help="PiP 거울모드 끄기(원본 방향). 기본은 거울처럼 좌우 반전")
    ap.add_argument("--no-display", action="store_true",
                    help="미리보기 창 끄기 (헤드리스 녹화/로깅)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="검증 N 프레임 후 종료 (0=무제한)")
    args = ap.parse_args()

    W, H = get_screen_size()
    print(f"모니터 해상도: {W}x{H}")
    print("3DGazeNet 로딩 중...")
    gazenet = GazeNetInference(0.5, args.det_size)
    reader = GazeReader(gazenet)
    stream = GazeStream(reader, buf_len=args.buf_len, angle_ema=args.angle_ema)
    print("모델 로드 완료.")

    # 녹화/세션 경로 결정
    record_path = args.record
    if args.session:
        sess = os.path.join(HERE, "recordings",
                            "session_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(sess, exist_ok=True)
        record_path = record_path or os.path.join(sess, "screen.mp4")
        if args.log is None:
            args.log = os.path.join(sess, "gaze.tsv")
        print(f"[session] 저장 폴더: {sess}")
    if record_path:
        os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
    recorder = _Recorder(record_path) if record_path else None

    cam = ZEDCamera(index=args.camera, resolution=args.resolution,
                    backend=args.backend, stereo=not args.mono, eye=args.eye)
    try:
        calib = None
        if args.recalibrate:
            print("[calib] --recalibrate: 저장된 보정 무시하고 처음부터 새로 보정합니다.")
        if args.load and not args.recalibrate and os.path.exists(CALIB_PATH):
            try:
                calib = json.load(open(CALIB_PATH, encoding="utf-8"))
            except Exception as e:
                print(f"[calib] 불러오기 실패({e}) -> 새로 보정")
                calib = None
            if calib is not None and calib.get("version") != CALIB_VERSION:
                print("[calib] 저장된 보정이 구버전 포맷 -> 호환 안 됨, 새로 보정 필요.")
                calib = None
            if calib is not None:
                calib["targets"] = [list(t) for t in calib.get("targets", [])]
                lp = calib.get("loo_px", calib.get("rms_px", 0.0))
                print(f"[calib] 불러옴: {CALIB_PATH} (LOO {lp:.0f}px, "
                      f"head {'ON' if calib.get('have_head') else 'OFF'})")
                # 저장 보정의 해상도와 현재 모니터 해상도가 다르면 좌표계 불일치 경고.
                if calib.get("W") != W or calib.get("H") != H:
                    print(f"[calib] 경고: 저장 해상도 {calib.get('W')}x{calib.get('H')} != "
                          f"현재 {W}x{H}. 좌표 매핑이 어긋날 수 있으니 'r'로 재보정 권장.")
        while True:
            if calib is None:
                calib = calibrate(cam, stream, W, H, args.points, args.ridge_lam)
                if calib is None:
                    print("보정 취소/실패.")
                    return 1
                os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
                json.dump(calib, open(CALIB_PATH, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
                print(f"[calib] 저장: {CALIB_PATH}")
            again = verify(cam, stream, calib, args, log_path=args.log,
                           recorder=recorder)
            if not again:
                break
            calib = None   # 'r' -> 재보정
    finally:
        cam.release()
        if recorder is not None:
            recorder.close()   # 예외/중단에도 녹화는 반드시 저장
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
