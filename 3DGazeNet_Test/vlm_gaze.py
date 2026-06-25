"""시선 × VLM 데모: 같은 얼굴 이미지를 3DGazeNet(정밀 수치)과 Gemini VLM(자연어 추정)에
각각 넣어 '어느 방향을 보는가'를 비교한다 (보고서 9.1절 a방향: VLM으로 시선 추정).

사용:
    # 공개 테스트 이미지 (외부전송 부담 적음)
    .venv/Scripts/python.exe vlm_gaze.py --image 3DGazeNet/demo/data/test_images/img1.jpg
    # 내 실제 얼굴(ZED 1프레임) — Gemini로 얼굴 전송됨, 동의 후 사용
    .venv/Scripts/python.exe vlm_gaze.py --camera

3DGazeNet은 정밀 3D 시선 벡터를, Gemini는 자연어 추정을 준다. 전용 모델 vs 범용 VLM 대비.
"""
import os, sys, math, argparse
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "3DGazeNet", "demo"))
from inference import GazeNetInference
GEMINI_MODEL = "gemini-2.5-flash"


def vec(r):
    g = r.get("gaze_out");  g = r.get("gaze") if g is None else g
    return np.asarray(g).ravel()[:3] if g is not None else np.array([np.nan]*3)


def yaw_pitch_label(g):
    gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
    # 3DGazeNet draw_gaze(utils.py:197)와 동일 규칙: 이미지상 화살표 변위 = (-gx, -gy).
    # 이미지(관찰자) 기준 x→오른쪽, y→아래 이므로 dx=-gx, dy=-gy 로 판정한다.
    dx, dy = -gx, -gy
    yaw = math.degrees(math.atan2(dx, abs(gz) + 1e-6))            # + → 오른쪽
    pitch = math.degrees(math.atan2(dy, math.hypot(gx, gz) + 1e-6))  # + → 아래
    h = "오른쪽" if yaw > 8 else ("왼쪽" if yaw < -8 else "정면")
    v = "아래" if pitch > 8 else ("위" if pitch < -8 else "정면")
    return yaw, pitch, f"{h}/{v}"


def ask_gemini(bgr_img):
    from google import genai
    import PIL.Image
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    pil = PIL.Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    prompt = (
        "이 사진에서 주된(가장 크게 보이는) 사람이 어느 방향을 보고 있는지 추정해 주세요. "
        "반드시 다음 형식으로만 답하세요:\n"
        "수평: 왼쪽|정면|오른쪽\n수직: 위|정면|아래\n"
        "추정각도: yaw=약 N도, pitch=약 N도 (대략)\n근거: 한 문장"
    )
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt, pil])
    return resp.text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(HERE, "3DGazeNet/demo/data/test_images/img1.jpg"))
    ap.add_argument("--camera", action="store_true", help="ZED에서 1프레임 캡처(얼굴이 Gemini로 전송됨)")
    ap.add_argument("--det-size", type=int, default=224)
    args = ap.parse_args()

    print("3DGazeNet 로딩...")
    gazenet = GazeNetInference(0.5, args.det_size)

    if args.camera:
        from zed_camera import ZEDCamera
        cam = ZEDCamera()
        img = out_gaze = out_img = None
        print("카메라에서 얼굴을 찾는 중... 정면을 봐주세요 (최대 ~6초).")
        for _ in range(90):                 # 얼굴 잡힐 때까지 프레임 탐색
            f = cam.read()
            if f is None:
                continue
            try:
                og, oi = gazenet.run(image=f, draw=True)
            except Exception:
                og, oi = [], f              # 얼굴 미검출 프레임은 건너뜀
            if og:
                img, out_gaze, out_img = f, og, oi
                break
        cam.release()
        if not out_gaze:
            print("얼굴을 못 찾았습니다. 카메라 앞 정면에 위치한 뒤 다시 실행해 주세요.")
            return 1
    else:
        img = cv2.imread(args.image)
        if img is None:
            print(f"이미지 읽기 실패: {args.image}"); return 1
        try:
            out_gaze, out_img = gazenet.run(image=img, draw=True)
        except Exception as e:
            print("얼굴 검출/추정 실패:", e); return 1
    out_path = os.path.join(HERE, "vlm_gaze_out.jpg")
    cv2.imwrite(out_path, out_img)

    print("\n================ 3DGazeNet (전용 모델, 정밀) ================")
    print(f"검출 얼굴 수: {len(out_gaze)}")
    for i, r in enumerate(out_gaze):
        g = vec(r); yaw, pitch, lab = yaw_pitch_label(g)
        print(f"  face[{i}] gaze=[{g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f}]  yaw≈{yaw:+.0f}° pitch≈{pitch:+.0f}°  → {lab}")
    print(f"(시각화 저장: {out_path})")

    print("\n================ Gemini VLM (범용, 자연어 추정) ================")
    try:
        print(ask_gemini(img))
    except Exception as e:
        print("Gemini 호출 실패:", type(e).__name__, str(e)[:300])
    print("\n* 좌우/상하 라벨 = '이미지(관찰자) 기준' (3DGazeNet draw_gaze 화살표와 동일 규칙: 변위 -gx,-gy). vlm_gaze_out.jpg 화살표와 일치.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
