# video_predict.py (updated process_video_file)
def process_video_file(uploaded_file, progress_cb=None):
    import tempfile, time, sys, os, subprocess
    import cv2
    import numpy as np
    # imageio-ffmpeg helper
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_exe = get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = None  # fallback to system ffmpeg if available

    # 1) Save uploaded file to temp input
    t_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        t_in.write(uploaded_file.read())
        t_in.flush()
    finally:
        t_in.close()

    # 2) Open input video
    cap = cv2.VideoCapture(t_in.name)
    if not cap.isOpened():
        try: os.remove(t_in.name)
        except: pass
        raise RuntimeError(f"Cannot open uploaded video file ({t_in.name}).")

    # 3) Read properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or (isinstance(fps, float) and np.isnan(fps)):
        fps = 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 4) Temp output (raw from OpenCV)
    t_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = t_out.name
    t_out.close()

    # 5) Use mp4v for writer; we'll transcode with ffmpeg to H264 later
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        try: os.remove(t_in.name)
        except: pass
        raise RuntimeError("Failed to open VideoWriter. Check codecs / OpenCV build.")

    # 6) Process frames
    frames_written = 0
    start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # run inference (YOLO expects BGR)
            try:
                res = model(frame)[0]
            except Exception as e:
                print("Warning: inference failed on frame:", e, file=sys.stderr)
                res = None

            if res is not None:
                # draw detections (safe handling)
                for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    try:
                        cls_idx = int(cls.item()) if hasattr(cls, "item") else int(cls)
                    except Exception:
                        cls_idx = int(cls)
                    label = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
                    is_carn = label.lower() in carnivores
                    color = (0,0,255) if is_carn else (0,255,0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {float(conf)*100:.0f}%", (x1, max(10, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                # optional overlay of counts per-frame could be added here

            writer.write(frame)
            frames_written += 1

            if progress_cb and total_frames:
                try:
                    progress_cb(min(frames_written / float(total_frames), 1.0))
                except Exception:
                    pass
    finally:
        cap.release()
        try: writer.release()
        except: pass

    elapsed = time.time() - start

    # 7) Transcode to browser-friendly mp4 (H.264 + yuv420p)
    final_t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    final_out = final_t.name
    final_t.close()

    # prefer bundled imageio-ffmpeg, else fall back to 'ffmpeg' on PATH
    ffexe = ffmpeg_exe or "ffmpeg"

    cmd = [
        ffexe,
        "-y",
        "-i", out_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-crf", "23",
        final_out
    ]

    try:
        # run ffmpeg and capture output for debugging if needed
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        # include stderr in raised error to help debug if ffmpeg missing or fails
        err = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        # cleanup temporary files
        try: os.remove(t_in.name)
        except: pass
        try: os.remove(out_path)
        except: pass
        raise RuntimeError(f"FFmpeg transcode failed: {err}")

    # 8) cleanup raw OpenCV file (we keep final_out)
    try:
        os.remove(out_path)
    except: pass
    try:
        os.remove(t_in.name)
    except: pass

    # 9) Verify final file looks OK
    try:
        size = os.path.getsize(final_out)
    except Exception:
        size = 0
    if size < 1024:
        raise RuntimeError(f"Output video seems invalid (size={size} bytes). Path: {final_out}")

    return final_out, elapsed, frames_written
