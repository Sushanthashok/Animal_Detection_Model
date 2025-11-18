🐾 Animal Detection Web App (My Custom Model + Streamlit)

This project is a fully custom-built animal detection system developed for my internship project.
I trained my own object detection model from scratch using a YOLO-based architecture and a custom dataset of 95+ animal species.

The system detects animals in images and videos, highlights carnivores in red, and generates fully annotated videos.

📘 Problem Statement

Wildlife monitoring teams face challenges such as:

Identifying animals automatically

Distinguishing carnivores from non-carnivores

Processing long videos frame-by-frame

Creating a simple UI for non-technical users

Playing processed videos directly in the browser

This project solves these problems using my custom-trained model and a Streamlit web interface.

📚 Dataset

For this internship project, I created and used a custom dataset containing 95 animal classes, including:

Elephant, Tiger, Lion

Rhino, Giraffe, Zebra

Fox, Wolf, Bear

Parrot, Flamingo, Penguin

And many more…

Dataset Notes

YOLO-format annotated images

Thousands of samples

Diverse lighting, angle, background conditions

Configured using my custom animal.yaml

⚙️ Methodology
### 🧠 Model Development (My Own Custom Model)

Architecture: YOLO-based object detector (custom configuration)

Trained using my dataset

Loss tuning & threshold optimization

Confidence filtering to prevent duplicate detections

Special mapping for carnivore identification

### 🎯 Carnivore Highlighting Logic

Carnivores → Red bounding box

Non-carnivores → Green bounding box

No extra yellow popups or confusion

### 🎥 Video Processing Steps

User uploads a video

Saved to a temporary location

Read frame-by-frame

My model detects animals on each frame

Draw bounding boxes + labels

Write a new annotated MP4 video using FFmpeg-supported codecs

Streamlit displays the final video + species summary

🧩 FFmpeg Requirement (Important for Annotated Video Playback)

To view the annotated processed video inside the Streamlit app,
FFmpeg must be installed.

✔ Check if FFmpeg is available:
ffmpeg -version


If the command is not found:

✔ Windows Installation:
winget install --id=Gyan.FFmpeg -e


After installation, restart your terminal, then check again:

ffmpeg -version


If FFmpeg is missing, the processed video will download, but will not play inside Streamlit.

🧪 Results
### ✔ Image Detection

Accurate single prediction per animal

NO duplicate classes on one object

Carnivores colored correctly in red

Clean bounding boxes without noise

### ✔ Video Detection

Smooth processing across frames

Annotated MP4 plays inside the app

Summary of species counts is auto-generated
