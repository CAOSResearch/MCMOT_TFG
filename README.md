# Real-Time Multi-Object Tracking using Intelligent Infrastructure Computer Vision (MCMOT_TFG)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/YOLO-11n-green.svg)](https://docs.ultralytics.com/)
[![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-orange.svg)](https://github.com/ifzhang/ByteTrack)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Official repository for the Bachelor's Thesis (**Trabajo de Fin de Grado - TFG**) titled *"Real-Time Multi-Object Tracking using Intelligent Infrastructure Computer Vision"*, developed within the **CAOSResearch** group and the **Safe4Car** project at **Universidad Carlos III de Madrid (UC3M)**.

---

## 📌 Overview

Maintaining identity continuity across **non-overlapping camera networks** is a critical challenge in Multi-Camera Multi-Object Tracking (MC-MOT), particularly when targets possess highly similar visual appearances (e.g., autonomous golf carts). 

This project delivers an end-to-end, modular tracking-by-detection framework that detects, tracks, and preserves the global identities of autonomous vehicles (specifically the **LSI iCab1** and **iCab2** golf carts) across non-overlapping roadside camera views. By combining custom object detection, single-camera multi-object tracking, deep appearance re-identification, and a topology-aware **Global Identity Manager (GIDM)**, the system achieves **96.9% cross-camera transition accuracy** and **100% occlusion recovery**.

```
                           +----------------------------------+
                           |       Physical Camera Setup      |
                           | (Non-Overlapping Basler Cameras) |
                           +-----------------+----------------+
                                             |
                   +-------------------------+-------------------------+
                   |                                                   |
                   v                                                   v
      [Camera 1 Stream (Left)]                            [Camera 2 Stream (Right)]
                   |                                                   |
                   v                                                   v
     +---------------------------+                       +---------------------------+
     | Object Detection (YOLO11n)|                       | Object Detection (YOLO11n)|
     +-------------+-------------+                       +-------------+-------------+
                   |                                                   |
                   v                                                   v
     +---------------------------+                       +---------------------------+
     | Single-Cam Track (ByteTrk)|                       | Single-Cam Track (ByteTrk)|
     +-------------+-------------+                       +-------------+-------------+
                   |                                                   |
                   v                                                   v
     +---------------------------+                       +---------------------------+
     | Appearance ReID (ResNet50)|                       | Appearance ReID (ResNet50)|
     +-------------+-------------+                       +-------------+-------------+
                   |                                                   |
                   +-------------------------+-------------------------+
                                             |
                                             v
                           +----------------------------------+
                           |      Global Identity Manager     |
                           | (Temporal & Appearance Fusion)   |
                           +----------------------------------+
```

---

## ✨ Key Features

* **Real-Time Object Detection**: Custom-trained **YOLO11n** model optimized for high inference speed and precision ($\sim 1.0$ precision/recall, $0.998$ $\text{mAP}_{50}$) on dedicated golf-cart detection.
* **Single-Camera Tracking (SCT)**: Integrated **ByteTrack** tracking algorithm with a 200-frame lost-track buffer to handle short occlusions and trajectory fluctuations.
* **Appearance Re-Identification**: Deep feature vector (embedding) extraction using **ResNet-50** backbones with L2 normalization and cosine similarity matching.
* **Topology-Aware Global Identity Manager (GIDM)**: 
  * Gaussian temporal transition modeling based on known camera network topology ($\mu = 110$ frames, $\sigma = 30$ frames).
  * Score fusion combining temporal consistency ($70\%$) and appearance similarity ($30\%$) to distinguish visually identical vehicles.
  * State-based identity lifecycle management (`Active`, `Lost`, `Reassigned`, `Removed`) with a 3-frame confirmation filter.
* **V2I / Safe4Car Integration**: Designed for cooperative intelligent transport infrastructure, serialized for Vehicle-to-Infrastructure (V2I) communication under ETSI standards (CAM / DENM).
* **Dual Execution Modes**: Native support for both offline video files and real-time **RTSP live camera streams**.

---

## 🏗️ System Architecture & Pipeline

The system processes multi-camera streams through four sequential stages:

1. **Object Detection Module**: Receives 2560×1440 image streams, resizes frames, and predicts bounding boxes using custom-trained `YOLO11n` (or `YOLOv8n`) weights.
2. **Single-Camera Multi-Object Tracking**: Applies ByteTrack to associate consecutive detections, generating coherent local trajectories (*tracklets*).
3. **Vehicle Re-Identification Module**: Crops detected vehicles and passes them through a ResNet-50 feature extractor to produce L2-normalized 2048-dimensional embeddings.
4. **Global Identity Manager**: Matches incoming local tracklets against active/lost global identity records using weighted fusion:
   $$S = w_a \cdot S_a + w_t \cdot S_t$$
   where $S_a$ is cosine similarity, $S_t = \exp\left(-\frac{(dt - \mu)^2}{2\sigma^2}\right)$ is the Gaussian temporal score, $w_a = 0.30$, and $w_t = 0.70$.

---

## 🛠️ Hardware & Infrastructure Setup

* **Sensing Infrastructure**: 10-meter roadside pole equipped with two **Basler ace 2 RGB cameras** (2560×1440 resolution @ 25 fps, non-overlapping FOV).
* **Computing Station**: NVIDIA A100 GPU server, 128 GB RAM, 1 TB SSD.
* **Test Vehicles**: Autonomous golf carts (*iCab1* and *iCab2*) operating in outdoor university corridors.

---

## 📂 Repository Structure

```
MCMOT_TFG/
├── models/                  # Pre-trained YOLOv8n/YOLO11n weights & ReID model checkpoints
│   └── custom_model.pt
├── dataset/
│   └── videos/              # Directory for input camera video streams
│       ├── video_left.mp4
│       └── video_right.mp4
├── pipeline.py              # Central pipeline script orchestrating all stages
├── .gitattributes
└── README.md
```

---

## 🚀 Installation & Prerequisites

### Prerequisites
* Python 3.10 or higher
* CUDA-compatible GPU (recommended for real-time inference)

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/CAOSResearch/MCMOT_TFG.git
   cd MCMOT_TFG
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install ultralytics torch torchvision opencv-python numpy scipy
   ```

---

## 💻 Usage

### 1. Offline Video Processing

1. Place your left and right camera video files inside `dataset/videos/`.
2. Configure the source paths, model weights, and output directory in `pipeline.py`:

   ```python
   SOURCES = {
       0: "dataset/videos/video_left.mp4",
       1: "dataset/videos/video_right.mp4"
   }
   
   OUTPUT_DIR = "output_results/"

   models = {
       cam_id: YOLO("models/custom_model.pt")
       for cam_id in SOURCES
   }
   ```

3. Run the tracking pipeline:
   ```bash
   python pipeline.py
   ```

---

### 2. Real-Time Deployment (RTSP Streams)

To deploy the framework in live monitoring scenarios using IP/industrial cameras over RTSP:

1. Update `SOURCES` in `pipeline.py` with your RTSP addresses:
   ```python
   SOURCES = {
       0: "rtsp://user:password@192.168.1.10/stream",
       1: "rtsp://user:password@192.168.1.11/stream"
   }
   ```

2. Execute the script:
   ```bash
   python pipeline.py
   ```

---

## ⚙️ Hyperparameter Configuration

The default hyperparameter configuration optimized for the Safe4Car test site is summarized below:

| Component | Parameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Object Detector** | Confidence Threshold | `0.40` | Minimum confidence score for valid detections |
| | IoU Threshold | `0.50` | NMS Intersection over Union threshold |
| **ByteTrack** | Lost-Track Buffer | `200 frames` | Retention time for temporary occlusions |
| | Track Activation | `0.50` | Minimum score to activate new tracklet |
| **Global ID Manager**| Expected Transition ($\mu$)| `110 frames` | Expected inter-camera transition (~4 seconds) |
| | Transition Sigma ($\sigma$) | `30 frames` | Temporal tolerance window |
| | Temporal Weight ($w_t$) | `0.70` | Weight assigned to temporal score |
| | Appearance Weight ($w_a$)| `0.30` | Weight assigned to cosine similarity |
| | Assignment Threshold | `0.60` | Minimum combined score for re-association |
| | Confirmation Hits | `3 frames` | Consecutive positive matches required |

---

## 📊 Experimental Results

Experimental evaluation on real-world test scenarios yielded the following quantitative performance metrics across different configurations:

| Configuration | Cross-Camera Transitions | Pedestrian Occlusions | Vehicle Interaction Occlusions |
| :--- | :---: | :---: | :---: |
| **Baseline System** | $0/32$ ($0.0\%$) | $2/5$ ($40.0\%$) | $0/8$ ($0.0\%$) |
| **Optimized Detection Parameters** | $0/32$ ($0.0\%$) | $5/5$ ($100.0\%$) | $6/8$ ($75.0\%$) |
| **GIDM + Temporal Restrictions** | **$31/32$ ($96.9\%$)** | **$5/5$ ($100.0\%$)** | **$8/8$ ($100.0\%$)** |

---

## 🎓 Academic Context & Citation

This software repository forms part of the Bachelor's Degree Thesis (**Trabajo de Fin de Grado**):

* **Title**: *Real-Time Multi-Object Tracking using Intelligent Infrastructure Computer Vision*
* **Author**: Pablo Molina Ponce de León ([@P-Molina-PL](https://github.com/P-Molina-PL))
* **Advisors**: David Yagüe Cuevas
* **Institution**: Universidad Carlos III de Madrid (UC3M) — Industrial Electronics and Automation Engineering (2025–2026)
* **Research Group / Project**: CAOSResearch / Safe4Car Project (PID2022-140554OB-C32)

If you use this codebase or methodology in your research, please cite:

```bibtex
@thesis{molina2026mcmot,
  author       = {Pablo Molina Ponce de León},
  title        = {Real-Time Multi-Object Tracking using Intelligent Infrastructure Computer Vision},
  school       = {Universidad Carlos III de Madrid},
  year         = {2026},
  type         = {Bachelor's Thesis},
  note         = {Safe4Car Project PID2022-140554OB-C32}
}
```

---

## 📄 License

This repository is licensed under the Creative Commons **Attribution - Non Commercial - Non Derivatives 4.0 International** ([CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)) license.
