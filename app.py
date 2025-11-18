# app.py (updated video processing + ffmpeg fallback)
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from collections import Counter
import shutil
import subprocess

st.set_page_config(page_title="Animal Detection App", page_icon="🐾", layout="wide")
st.title("🐾 Animal Detection App")
st.write("Upload an image or video to detect animals using your YOLO model. Carnivores are highlighted in red.")

# ---------------------------
# Config (edit these if needed)
# ---------------------------
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.35
IOU_FILTER = 0.45
VIDEO_CODEC = "XVID"   # writer fallback (we will transcode to H.264 with ffmpeg if available)
FFMPEG_CMD = shutil.which("ffmpeg")  # path to ffmpeg if available

# class names and carnivores (same as your training)
class_names = ['Balaeniceps rex', 'Cybister', 'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar',
               'buffalo', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'crab', 'crow', 'deer',
               'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
               'giraffe', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog',
               'hippopotamus', 'hornbill', 'horse', 'human', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala',
               'ladybug', 'leopard', 'lion', 'lizard', 'lobster', 'mammoth', 'mosquito', 'moth', 'mouse', 'octopus',
               'okapi', 'opossum', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes',
               'penguin', 'pig', 'pigeon', 'porcupine', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper',
               'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan',
               'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

CARNIVORES = {'bear','cat','chimpanzee','dog','eagle','fox','gorilla','hyena','leopard','lion','opossum',
              'orangutan','owl','raccoon','rat','shark','snake','tiger','wolf'}


# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)
st.success("Model loaded", icon="✅")


# ---------------------------
# small helpers
# ---------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    areaB = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    union = areaA + areaB - inter
    return inter/union if union > 0 else 0.0

def merge_detections(boxes, classes, confs, iou_thresh=IOU_FILTER):
    items = list(zip(boxes, classes, confs))
    items.sort(key=lambda x: float(x[2]), reverse=True)
    kept = []
    for b, c, s in items:
        skip = False
        for kb, kc, ks in kept:
            if iou(b, kb) > iou_thresh:
                skip = True
                break
        if not skip:
            kept.append((b, c, s))
    if not kept:
        return [], [], []
    boxes_k, classes_k, confs_k = zip(*kept)
    return list(boxes_k), list(classes_k), list(confs_k)

def draw_and_annotate(frame_bgr, boxes, classes_idx, confs):
    out = frame_bgr.copy()
    carn_count = 0
    for (x1, y1, x2, y2), cls_idx, conf in zip(boxes, classes_idx, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
        is_carn = label.lower() in CARNIVORES
        color = (0,0,255) if is_carn else (0,200,0)   # red for carnivore, green otherwise
        if is_carn: carn_count += 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{label} {float(conf):.2f}", (x1, max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out, carn_count


# ---------------------------
# Process image (unchanged logic)
# ---------------------------
def process_image_pil(pil_image):
    img = np.array(pil_image.convert("RGB"))[:, :, ::-1]  # RGB->BGR
    res = model(img, conf=CONF_THRESHOLD)[0]
    boxes = [tuple(map(int, b.tolist())) for b in (res.boxes.xyxy.cpu().numpy() if len(res.boxes) else [])]
    classes_idx = [int(c) for c in (res.boxes.cls.cpu().numpy() if len(res.boxes) else [])]
    confs = [float(c) for c in (res.boxes.conf.cpu().numpy() if len(res.boxes) else [])]

    filt = [(b, ci, cf) for b, ci, cf in zip(boxes, classes_idx, confs) if cf >= CONF_THRESHOLD]
    if filt:
        boxes, classes_idx, confs = zip(*filt)
        boxes, classes_idx, confs = list(boxes), list(classes_idx), list(confs)
        boxes, classes_idx, confs = merge_detections(boxes, classes_idx, confs)
    else:
        boxes, classes_idx, confs = [], [], []

    annotated, carn_count = draw_and_annotate(img, boxes, classes_idx, confs)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    detected = [f"{class_names[c]} ({conf:.2f})" for c, conf in zip(classes_idx, confs)]
    return annotated_rgb, detected, carn_count


# ---------------------------
# Process video (robust)
# ---------------------------
def process_video_file(uploaded_file, progress_cb=None):
    """
    Returns (final_out_path, frames_written, counts, elapsed_seconds).
    If ffmpeg is found in PATH, we transcode the writer output to H.264 MP4 (browser-friendly).
    """
    import tempfile, os, time
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp_in.write(uploaded_file.read())
        tmp_in.flush()
    finally:
        tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        try: os.remove(tmp_in.name)
        except: pass
        raise RuntimeError("Cannot open uploaded video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # write an intermediate file (AVI with XVID, broadly supported by OpenCV)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
    writer_path = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(writer_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        try: os.remove(tmp_in.name)
        except: pass
        raise RuntimeError("Failed to open VideoWriter. Install ffmpeg and restart terminal to enable MP4/H264 transcoding.")

    frames_written = 0
    counts = Counter()
    start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                res = model(frame, conf=CONF_THRESHOLD)[0]
            except Exception as e:
                # write original frame and continue
                writer.write(frame)
                frames_written += 1
                if progress_cb and total_frames:
                    progress_cb(min(frames_written/total_frames, 1.0))
                continue

            boxes = [tuple(map(int, b.tolist())) for b in (res.boxes.xyxy.cpu().numpy() if len(res.boxes) else [])]
            classes_idx = [int(c) for c in (res.boxes.cls.cpu().numpy() if len(res.boxes) else [])]
            confs = [float(c) for c in (res.boxes.conf.cpu().numpy() if len(res.boxes) else [])]

            filt = [(b, ci, cf) for b, ci, cf in zip(boxes, classes_idx, confs) if cf >= CONF_THRESHOLD]
            if filt:
                boxes, classes_idx, confs = zip(*filt)
                boxes, classes_idx, confs = list(boxes), list(classes_idx), list(confs)
                boxes, classes_idx, confs = merge_detections(boxes, classes_idx, confs)
            else:
                boxes, classes_idx, confs = [], [], []

            annotated, carn_count = draw_and_annotate(frame, boxes, classes_idx, confs)
            for ci in classes_idx:
                if 0 <= ci < len(class_names):
                    counts[class_names[ci]] += 1

            writer.write(annotated)
            frames_written += 1

            if progress_cb and total_frames:
                progress_cb(min(frames_written / max(total_frames, 1), 1.0))

    finally:
        cap.release()
        try:
            writer.release()
        except:
            pass

    elapsed = time.time() - start

    # If ffmpeg available -> transcode to H.264 MP4 (browser-friendly)
    final_path = writer_path
    if FFMPEG_CMD:
        final_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        final_tmp.close()
        final_path = final_tmp.name
        # build ffmpeg command (overwrite quietly)
        ff_cmd = [
            FFMPEG_CMD, "-y", "-i", writer_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            final_path
        ]
        try:
            subprocess.run(ff_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            # fallback: use the writer_path itself if ffmpeg fails
            final_path = writer_path
    else:
        # ffmpeg not found — warn caller later (final_path remains writer_path)
        pass

    # cleanup input .mp4 (we can remove original uploaded tmp)
    try:
        os.remove(tmp_in.name)
    except:
        pass

    # ensure final file is okay
    try:
        size = os.path.getsize(final_path)
    except:
        size = 0
    if size < 1024:
        raise RuntimeError("Processed video file appears invalid or empty. Try installing ffmpeg and restart the shell.")

    # optionally remove the intermediate writer file if we transcoded
    if final_path != writer_path:
        try: os.remove(writer_path)
        except: pass

    return final_path, frames_written, counts, elapsed


# ---------------------------
# Streamlit UI
# ---------------------------
tab1, tab2 = st.tabs(["Image", "Video"])

with tab1:
    st.header("Image detection")
    uploaded = st.file_uploader("Upload an image (jpg,jpeg,png)", type=["jpg","jpeg","png"])
    if uploaded:
        pil = Image.open(uploaded)
        st.subheader("Original")
        st.image(pil, use_container_width=True)

        if st.button("Run detection on image"):
            with st.spinner("Detecting..."):
                annotated_rgb, detected_list, carn_count = process_image_pil(pil)
            st.subheader("Annotated")
            st.image(annotated_rgb, use_container_width=True)

            st.markdown("### Detected animals")
            if detected_list:
                for d in detected_list:
                    st.write("•", d)
            else:
                st.write("No animals detected.")
            # show carnivore count as plain text (no popup)
            st.info(f"Carnivores in image: {carn_count}")

with tab2:
    st.header("Video detection")
    vfile = st.file_uploader("Upload a video (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"])
    if vfile:
        st.subheader("Original")
        st.video(vfile)

        if st.button("Process video"):
            progress = st.progress(0.0)
            status = st.empty()
            try:
                def cb(fr):
                    progress.progress(min(max(fr, 0.0), 1.0))
                    status.text(f"Processed {int(fr*100)}%")
                out_path, frames_written, counts, elapsed = process_video_file(vfile, progress_cb=cb)
                progress.progress(1.0)
                status.text(f"Done — {frames_written} frames in {elapsed:.1f}s")
            except Exception as e:
                st.error(f"Video processing failed: {e}")
                if not FFMPEG_CMD:
                    st.warning("ffmpeg not found in PATH — install ffmpeg and restart your terminal/VS Code to produce browser-friendly MP4.")
                raise

            # load result bytes and play inside app
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.subheader("Processed (annotated) video")
            st.video(video_bytes)

            st.markdown("### 🔍 Detected animals (summary for whole video)")
            if counts:
                for name, cnt in counts.most_common():
                    st.write(f"• **{name}** : {cnt}")
            else:
                st.write("No animals detected in video.")

            st.download_button("Download annotated video", data=video_bytes, file_name="annotated_video.mp4", mime="video/mp4")

            # cleanup final
            try: os.remove(out_path)
            except: pass

    # helpful note about ffmpeg
    if not FFMPEG_CMD:
        st.info("Tip: For best browser playback install ffmpeg and restart your terminal/VS Code (you already tried 'winget install ...'). After restart Streamlit will detect ffmpeg and create browser-friendly H.264 mp4 files.")


