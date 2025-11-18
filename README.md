# 🐾 Animal Detection Web App (Custom Model + Streamlit)

This project is a fully custom-built animal detection system, developed as part of my internship project.
I trained my own YOLO-based object detection model from scratch on a custom dataset of 95+ animal species.

The system can:

Detect animals in images & videos

Highlight carnivores in red

Generate annotated videos

Provide a summary of detected species

 ## 📘 Problem Statement

Wildlife monitoring teams face challenges such as:

🔍 Automatically identifying animals

🐺 Distinguishing carnivores vs non-carnivores

🎞 Processing long videos frame-by-frame

🧑‍💻 Providing an easy UI for non-technical users

▶️ Playing annotated output videos inside a web app

This project solves these issues using my custom-trained detection model + Streamlit interface.

## 📚 Dataset Description

For this internship project, I created and used a custom dataset with 95 animal classes such as:

🐘 Elephant

🐅 Tiger

🦁 Lion

🦏 Rhino

🦒 Giraffe

🦓 Zebra

🦊 Fox

🐺 Wolf

🐻 Bear

🐧 Penguin

🦩 Flamingo

🦜 Parrot

### Dataset Notes

📁 YOLO-format labeled images

📸 Thousands of samples

🌤 Wide variations: lighting, angles, camera types

⚙️ Configured through a custom animal.yaml

## ⚙️ Methodology

### 🧠 1. Custom Model Development

YOLO-based architecture (custom configuration)

Trained on my own dataset

Loss tuning and hyperparameter optimization

Confidence thresholding to avoid duplicate predictions

Custom mapping for identifying carnivores

### 🎯 2. Carnivore Highlighting Logic

🔴 Carnivores → Red bounding box

🟢 Non-carnivores → Green bounding box

❌ No yellow pop-ups or noisy overlays

### 🎥 3. Video Processing Pipeline

User uploads a video

Saved to temporary storage

Read frame-by-frame

My model detects animals

Bounding boxes + labels drawn

Video re-encoded using FFmpeg-compatible codec

Streamlit displays the annotated video + summary

### 🧩 FFmpeg Requirement (Important)

To play annotated MP4 videos inside Streamlit, FFmpeg must be installed.

✅ Check FFmpeg:
ffmpeg -version

❌ If it says “command not found”, install using:
winget install --id=Gyan.FFmpeg -e


Then restart terminal:

ffmpeg -version


### ⚠️ Without FFmpeg:

Video will download,

But will NOT play inside the Streamlit app.

## Make sure run this before run your app.py

This does not require restarting VS Code or Windows. Run the following PowerShell block exactly — it searches for the installed ffmpeg.exe, adds its folder to the session PATH, and then shows version info:
```
# find ffmpeg.exe and add its folder to this shell's PATH (one-shot, session only)
$ff = Get-ChildItem -Path $env:LOCALAPPDATA -Filter 'ffmpeg.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ff) {
    $folder = $ff.DirectoryName
    $env:PATH = "$env:PATH;$folder"
    Write-Output "Added to PATH for this session: $folder"
    ffmpeg -version
} else {
    Write-Error "ffmpeg.exe not found under $env:LOCALAPPDATA. If not found, close & re-open your terminal (Option 2)."
}
```
## 🧪 Results
### ✔️ Image Detection

🟢 Clean bounding boxes

🚫 No duplicate detections

🔴 Carnivores correctly detected in red

📌 High confidence predictions

### ✔️ Video Detection

🎞 Smooth frame-by-frame annotation

🖼 Output MP4 plays directly in Streamlit

📊 Animal count summary auto-generated

## 📁 Project Structure
```
project/
 ├── app.py
 ├── video_predict.py
 ├── image_predict.py
 ├── animal.yaml
 ├── check_labels.py
 ├── requirements.txt
 ├── best.pt         ← (MY OWN TRAINED MODEL)
 ├── runs/                       
 └── data/
```            

## DATA,VISUAL OUTPUT and TRAINING FILE LINK

[Drive Link](https://drive.google.com/drive/folders/142eIYXz-W3-OQKzkgsFblZ_kjQyBlGrn?usp=sharing)
