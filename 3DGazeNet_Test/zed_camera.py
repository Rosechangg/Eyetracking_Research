"""ZED 카메라 프레임 소스 (3DGazeNet 연동용).

3DGazeNet은 단안(monocular) RGB gaze 추정기라 depth가 필요 없다. ZED는 USB로
좌/우가 가로로 붙은(side-by-side) 스테레오 영상을 일반 UVC 카메라처럼 내보내므로,
ZED SDK(pyzed) 없이도 OpenCV로 받아 **왼쪽 절반만** 떼어 쓰면 된다.

두 가지 백엔드:
  - "opencv" (기본): cv2.VideoCapture로 스테레오 프레임을 받아 왼쪽 눈 영상만 사용.
  - "pyzed": ZED SDK가 설치돼 있으면 sl.Camera로 왼쪽 RGB를 직접 사용.

ZED USB 스테레오 해상도 (전체 side-by-side 폭 x 높이):
    HD2K  : 4416 x 1242   (eye 2208 x 1242)
    HD1080: 3840 x 1080   (eye 1920 x 1080)
    HD720 : 2560 x 720    (eye 1280 x 720)
    VGA   : 1344 x 376    (eye 672 x 376)
각 눈이 16:9라 side-by-side 프레임의 종횡비는 약 32:9 (= 3.556). 이게 ZED 자동 탐지 신호다.
"""
import sys
import cv2
import numpy as np

ZED_STEREO_MODES = {
    "HD2K":   (4416, 1242),
    "HD1080": (3840, 1080),
    "HD720":  (2560, 720),
    "VGA":    (1344, 376),
}

_STEREO_ASPECT = 32.0 / 9.0   # 3.556, side-by-side 두 눈
_ASPECT_TOL = 0.25


def _aspect(w, h):
    return float(w) / float(h) if h else 0.0


class ZEDCamera:
    """ZED를 단일-눈 RGB 프레임 소스로 노출한다. read()는 BGR uint8 프레임 또는 None."""

    def __init__(self, index=None, resolution="HD720", backend="auto",
                 stereo=True, eye="left", api=None):
        self.resolution = resolution
        self.eye = eye
        self.stereo = stereo
        self.backend = backend
        self._zed = None          # pyzed Camera
        self._cap = None          # cv2.VideoCapture
        self._zed_mat = None

        if backend in ("auto", "pyzed"):
            if self._try_pyzed():
                self.backend = "pyzed"
                return
            if backend == "pyzed":
                raise RuntimeError("pyzed(ZED SDK) backend을 요청했지만 사용할 수 없습니다.")
        # OpenCV 백엔드
        self.backend = "opencv"
        self._open_opencv(index, api)

    # ---------------- pyzed ----------------
    def _try_pyzed(self):
        try:
            import pyzed.sl as sl
        except Exception:
            return False
        try:
            cam = sl.Camera()
            init = sl.InitParameters()
            res_map = {
                "HD2K": sl.RESOLUTION.HD2K,
                "HD1080": sl.RESOLUTION.HD1080,
                "HD720": sl.RESOLUTION.HD720,
                "VGA": sl.RESOLUTION.VGA,
            }
            init.camera_resolution = res_map.get(self.resolution, sl.RESOLUTION.HD720)
            if cam.open(init) != sl.ERROR_CODE.SUCCESS:
                return False
            self._sl = sl
            self._zed = cam
            self._zed_mat = sl.Mat()
            self._zed_view = sl.VIEW.LEFT if self.eye == "left" else sl.VIEW.RIGHT
            self._runtime = sl.RuntimeParameters()
            return True
        except Exception:
            return False

    # ---------------- opencv ----------------
    def _open_opencv(self, index, api):
        if api is None:
            # Windows: MSMF가 고해상도 설정이 잘 먹는 편. 안 되면 DSHOW로 폴백.
            api = cv2.CAP_MSMF if sys.platform.startswith("win") else cv2.CAP_ANY

        if index is None:
            index = self._autodetect(api)
            if index is None:
                # 스테레오 탐지 실패 → 0번을 mono로
                print("[ZED] 스테레오(32:9) 카메라 자동 탐지 실패 → index 0를 mono로 사용.")
                index = 0
                self.stereo = False

        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened() and api != cv2.CAP_ANY:
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not cap.isOpened():
            raise RuntimeError(f"카메라 index {index}를 열 수 없습니다.")

        if self.stereo and self.resolution in ZED_STEREO_MODES:
            w, h = ZED_STEREO_MODES[self.resolution]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        self._index = index
        # 실제 프레임이 32:9면 스테레오로 확정, 아니면 mono 처리
        if abs(_aspect(aw, ah) - _STEREO_ASPECT) <= _ASPECT_TOL:
            self.stereo = True
        print(f"[ZED] opencv backend: index={index} frame={aw}x{ah} "
              f"stereo={self.stereo} eye={self.eye}")

    def _autodetect(self, api, max_index=6):
        found = None
        for i in range(max_index):
            cap = cv2.VideoCapture(i, api)
            if not cap.isOpened():
                cap.release()
                continue
            # 스테레오 모드로 폭을 키워보고 종횡비 확인
            w, h = ZED_STEREO_MODES.get(self.resolution, (2560, 720))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, frame = cap.read()
            if ok and frame is not None:
                fh, fw = frame.shape[:2]
                if abs(_aspect(fw, fh) - _STEREO_ASPECT) <= _ASPECT_TOL:
                    found = i
                    cap.release()
                    print(f"[ZED] 자동 탐지: index {i} ({fw}x{fh}, 32:9 스테레오)")
                    break
            cap.release()
        return found

    # ---------------- common ----------------
    def read(self):
        if self.backend == "pyzed":
            if self._zed.grab(self._runtime) != self._sl.ERROR_CODE.SUCCESS:
                return None
            self._zed.retrieve_image(self._zed_mat, self._zed_view)
            bgra = self._zed_mat.get_data()
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        # opencv
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if self.stereo:
            half = frame.shape[1] // 2
            frame = frame[:, :half] if self.eye == "left" else frame[:, half:]
        return np.ascontiguousarray(frame)

    def release(self):
        if self._cap is not None:
            self._cap.release()
        if self._zed is not None:
            try:
                self._zed.close()
            except Exception:
                pass
