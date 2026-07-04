# 🏗️ Construction Site Safety Surveillance System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

## 📌 Overview

A real-time **Construction Site Safety Surveillance System** built using **YOLOv8** and **Streamlit**. The system detects Personal Protective Equipment (PPE) violations on construction sites from multiple input sources — images, videos, webcam, and YouTube URLs.

---

## 📊 Dataset

- **Source:** Real-world construction site footage from college campus construction site
- **Type:** Custom annotated dataset
- **Labels:** Helmet, No Helmet, Safety Vest, No Safety Vest, Gloves, No Gloves, Safety Boots, Person

---

## 🔬 Features

| Feature | Description |
|---|---|
| 📷 **Image Detection** | Upload any image for instant PPE violation detection |
| 🎬 **Video Detection** | Upload video files for frame-by-frame analysis |
| 📡 **Webcam Detection** | Real-time live detection from webcam |
| ▶️ **YouTube Detection** | Paste any YouTube URL for detection |
| ⚠️ **Alert System** | Automatic alerts for PPE violations |
| 🎚️ **Confidence Control** | Adjustable detection confidence threshold |

---

## 🛡️ PPE Classes Detected

| Class | Type |
|---|---|
| 🟢 Helmet | Safe |
| 🔴 No Helmet | **Violation** |
| 🟢 Safety Vest | Safe |
| 🔴 No Safety Vest | **Violation** |
| 🟢 Gloves | Safe |
| 🔴 No Gloves | **Violation** |
| 🟢 Safety Boots | Safe |
| 🟠 Person | Detected |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

**Libraries Used:**
- `ultralytics` — YOLOv8 model training and inference
- `streamlit` — Web application interface
- `opencv-python` — Image and video processing
- `yt-dlp` — YouTube video downloading
- `torch`, `torchvision` — Deep learning framework
- `Pillow` — Image handling

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/SowmyaGooty/construction-site-safety.git
cd construction-site-safety
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Add your trained YOLOv8 weights**
```bash
# Place your trained model file as:
best.pt
```

4. **Run the app**
```bash
streamlit run app.py
```

5. **Open in browser**
```
http://localhost:8501
```

---

## 📁 Project Structure

```
construction-site-safety/
│
├── app.py              # Main Streamlit application
├── best.pt             # YOLOv8 trained weights (add your own)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🎯 Key Features

- ✅ **Real-world dataset** from college campus construction site
- ✅ **4 detection modes** — image, video, webcam, YouTube
- ✅ **Real-time detection** with live webcam feed
- ✅ **Automatic violation alerts** with visual bounding boxes
- ✅ **Adjustable confidence threshold** via sidebar
- ✅ **Color-coded detections** — green for safe, red for violations

---

## 👩‍💻 Author

**Sowmya Gooty**
MS Engineering Data Science @ University of Houston
📧 sgooty@cougarnet.uh.edu
🔗 [LinkedIn](https://linkedin.com/in/sowmyagooty) | [GitHub](https://github.com/SowmyaGooty)

---

⭐ If you found this project useful, please consider giving it a star!
