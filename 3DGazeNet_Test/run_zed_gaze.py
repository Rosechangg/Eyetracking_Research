"""ZED 카메라 + 3DGazeNet 실시간 시선 추정 + 세션 녹화/데이터 저장 러너.

ZED에서 프레임을 받아(왼쪽 눈) 3DGazeNet으로 얼굴별 3D gaze 벡터를 추정하고,
화면에 시선/홍채를 그려 실시간 표시하면서, 동시에
  - 시각화 영상(.mp4)
  - 프레임별 gaze 데이터(.tsv: 타임스탬프 / gaze 벡터 / 홍채 중심)
를 저장한다. 'q' 또는 ESC로 종료.

사용 예:
    # 라이브 시각화 + 세션 자동 저장(recordings/session_<시각>/ 에 video.mp4 + gaze.tsv)
    .venv/Scripts/python.exe run_zed_gaze.py --session

    # 경로를 직접 지정
    .venv/Scripts/python.exe run_zed_gaze.py --record out.mp4 --save-gaze out.tsv

    # 시각화만(저장 X)
    .venv/Scripts/python.exe run_zed_gaze.py

    # 스모크 테스트(이미지 1장)
    .venv/Scripts/python.exe run_zed_gaze.py --image 3DGazeNet/demo/data/test_images/img1.jpg
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # matplotlib GUI 백엔드 회피
import sys
import time
import argparse
from datetime import datetime

# Windows 콘솔(cp949)에서 한글 출력이 죽지 않도록 UTF-8 강제
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

from inference import GazeNetInference          # noqa: E402  (3DGazeNet demo)
from zed_camera import ZEDCamera, ZED_STEREO_MODES  # noqa: E402


def _vec3(result):
    g = result.get("gaze_out")
    if g is None:
        g = result.get("gaze")
    return np.asarray(g).ravel()[:3] if g is not None else np.array([np.nan] * 3)


def _iris(result, side):
    ci = result.get("centers_iris")
    if isinstance(ci, dict) and ci.get(side) is not None:
        p = np.asarray(ci[side]).ravel()
        return float(p[0]), float(p[1])
    return float("nan"), float("nan")


def run_image(gazenet, path):
    img = cv2.imread(path)
    if img is None:
        print(f"이미지를 읽을 수 없음: {path}")
        return 1
    out_gaze, out_img = gazenet.run(image=img, draw=True)
    print(f"검출된 얼굴 수: {len(out_gaze)}")
    for i, r in enumerate(out_gaze):
        g = _vec3(r)
        print(f"  face[{i}] gaze=[{g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}]")
    out_path = os.path.join(HERE, "out_gaze.jpg")
    cv2.imwrite(out_path, out_img)
    print(f"결과 저장: {out_path}")
    return 0


class _Recorder:
    """시각화 프레임을 .mp4로 저장. 처음 몇 프레임으로 실제 FPS를 추정해 재생속도를 맞춘다."""
    def __init__(self, path, fps=None):
        self.path = path
        self.fps = fps          # 명시되면 측정 생략
        self.writer = None
        self._buf = []          # writer 열기 전 임시 버퍼
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
        print(f"[rec] 녹화 시작: {self.path} (~{fps:.1f} fps)")

    def close(self):
        if self.writer is None and self._buf:          # 짧게 끝난 경우
            fps = self.fps or (len(self._buf) / max(0.5, time.time() - (self._t0 or time.time())))
            self._open(fps)
        if self.writer is not None:
            self.writer.release()
            print(f"[rec] 녹화 저장 완료: {self.path}")


def run_live(gazenet, args):
    # 저장 경로 결정
    record_path, gaze_path = args.record, args.save_gaze
    if args.session:
        sess = os.path.join(HERE, "recordings",
                            "session_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(sess, exist_ok=True)
        record_path = record_path or os.path.join(sess, "video.mp4")
        gaze_path = gaze_path or os.path.join(sess, "gaze.tsv")
        print(f"[session] 저장 폴더: {sess}")
    for p in (record_path, gaze_path):
        if p:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    cam = ZEDCamera(index=args.camera, resolution=args.resolution,
                    backend=args.backend, stereo=not args.mono, eye=args.eye)

    gaze_log = open(gaze_path, "w", encoding="utf-8") if gaze_path else None
    if gaze_log:
        gaze_log.write("timestamp\tframe_idx\tface_idx\tgx\tgy\tgz"
                       "\tiris_left_x\tiris_left_y\tiris_right_x\tiris_right_y\n")
    recorder = _Recorder(record_path, fps=args.fps) if record_path else None

    win = "ZED + 3DGazeNet (q=quit)"
    if not args.no_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps, t_prev, frame_idx = 0.0, time.time(), 0
    print("실행 중... 종료하려면 미리보기 창에서 'q' (또는 ESC).")
    try:
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                break
            frame = cam.read()
            if frame is None:
                print("프레임 수신 실패 (카메라 분리/종료?)")
                break

            try:
                out_gaze, out_img = gazenet.run(image=frame, draw=True)
            except Exception:
                out_gaze, out_img = [], frame   # 얼굴 미검출 등

            ts = time.time()
            for fi, r in enumerate(out_gaze):
                if gaze_log is not None:
                    g = _vec3(r)
                    lx, ly = _iris(r, "left")
                    rx, ry = _iris(r, "right")
                    gaze_log.write(f"{ts:.6f}\t{frame_idx}\t{fi}"
                                   f"\t{g[0]:.5f}\t{g[1]:.5f}\t{g[2]:.5f}"
                                   f"\t{lx:.2f}\t{ly:.2f}\t{rx:.2f}\t{ry:.2f}\n")

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
            rec_tag = "  REC" if recorder else ""
            cv2.putText(out_img, f"FPS {fps:4.1f}  faces {len(out_gaze)}{rec_tag}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if recorder is not None:
                recorder.add(out_img.copy())
            if not args.no_display:
                cv2.imshow(win, out_img)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            frame_idx += 1
    finally:
        cam.release()
        if gaze_log is not None:
            gaze_log.close()
            print(f"[data] gaze 데이터 저장: {gaze_path}  ({frame_idx} 프레임)")
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()
    return 0


def main():
    ap = argparse.ArgumentParser(description="ZED + 3DGazeNet 실시간 시선 추정 + 녹화/저장")
    ap.add_argument("--camera", type=int, default=None, help="카메라 인덱스 (기본: ZED 자동 탐지)")
    ap.add_argument("--resolution", default="HD720", choices=list(ZED_STEREO_MODES.keys()),
                    help="ZED 스테레오 해상도 (기본 HD720)")
    ap.add_argument("--backend", default="auto", choices=["auto", "opencv", "pyzed"],
                    help="프레임 소스 백엔드 (기본 auto)")
    ap.add_argument("--eye", default="left", choices=["left", "right"], help="사용할 눈 (기본 left)")
    ap.add_argument("--mono", action="store_true", help="비스테레오(일반 웹캠)로 취급")
    ap.add_argument("--det-size", type=int, default=224, help="얼굴 검출 입력 (640=정확/느림, 224=빠름)")
    ap.add_argument("--det-thresh", type=float, default=0.5, help="얼굴 검출 임계값")
    # 저장 관련
    ap.add_argument("--session", action="store_true",
                    help="recordings/session_<시각>/ 에 video.mp4 + gaze.tsv 자동 저장")
    ap.add_argument("--record", default=None, metavar="PATH.mp4", help="시각화 영상 녹화 경로")
    ap.add_argument("--save-gaze", default=None, metavar="PATH.tsv", help="프레임별 gaze 데이터 저장 경로")
    ap.add_argument("--fps", type=float, default=None,
                    help="녹화 FPS 고정값 (미지정 시 실제 처리속도로 자동 추정)")
    # 기타
    ap.add_argument("--no-display", action="store_true", help="미리보기 창 끄기 (헤드리스 저장용)")
    ap.add_argument("--max-frames", type=int, default=0, help="N 프레임 후 종료 (0=무제한)")
    ap.add_argument("--image", default=None, help="카메라 대신 이미지 1장 추론(스모크 테스트)")
    args = ap.parse_args()

    print("3DGazeNet 로딩 중...")
    gazenet = GazeNetInference(args.det_thresh, args.det_size)
    print("모델 로드 완료.")

    if args.image:
        return run_image(gazenet, args.image)
    return run_live(gazenet, args)


if __name__ == "__main__":
    raise SystemExit(main())
