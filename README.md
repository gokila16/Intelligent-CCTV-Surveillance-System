# Intelligent CCTV Surveillance System

**Intelligent CCTV Surveillance System** is a deep-learning video analytics project for identifying suspicious activity and potential threats in surveillance footage. It combines frame-level object detection with temporal activity recognition to analyze both objects and behavior across video streams.

The project was developed by **Gokila Harini** and was presented as a research paper at the **Second International Conference on Next Generation Computing Systems (ICNGCS 2023)**.

## Project Overview

Traditional CCTV systems primarily record video for later review. This project explores how computer vision can support real-time monitoring by automatically identifying weapons and recognizing potentially violent activity.

The solution brings together two complementary deep-learning workflows:

- **YOLOv7-based object detection** for identifying weapons such as knives and pistols.
- **CNN-LSTM activity recognition** for analyzing sequences of video frames and classifying activity as `fight` or `noFight`.

## Key Highlights

- Achieved **92% detection accuracy** across evaluated multi-class surveillance scenarios.
- Improved video-processing throughput by **2.3×** through parallel processing of multiple video feeds.
- Strengthened performance in low-light conditions using histogram equalization, noise filtering, and spatial smoothing.
- Improved low-light robustness and signal quality by **35%** during evaluation.
- Built and annotated a diverse surveillance-event dataset.
- Applied data augmentation, hyperparameter tuning, and cross-validation to improve model generalization.
- Supported live webcam-based activity prediction through Google Colab.

## System Workflow

```text
Video Feed
    │
    ├── Frame Preprocessing
    │       ├── Noise Filtering
    │       ├── Spatial Smoothing
    │       └── Histogram Equalization
    │
    ├── YOLOv7 Weapon Detection
    │       └── Knife / Pistol Detection
    │
    └── CNN-LSTM Activity Recognition
            └── Fight / No-Fight Classification
```

## Repository Contents

| File | Description |
| --- | --- |
| `CNN+LSTM_Activity_detection_in_CCTV.ipynb` | Video activity-recognition workflow using a CNN-LSTM/LRCN-style model. |
| `weapon_detection_final.ipynb` | Custom YOLOv7 training and inference workflow for weapon detection. |

## Technology Stack

- Python
- TensorFlow and Keras
- PyTorch and TorchVision
- OpenCV
- YOLOv7
- CNN-LSTM / LRCN-style temporal modeling
- NumPy, Matplotlib, and scikit-learn
- Roboflow for dataset preparation and export
- Google Colab with GPU acceleration

## Running the Project

The notebooks are configured for Google Colab and should be run with GPU acceleration enabled.

### Activity Recognition

1. Open `CNN+LSTM_Activity_detection_in_CCTV.ipynb` in Google Colab.
2. Mount Google Drive when prompted.
3. Place the activity dataset in the following structure:

   ```text
   MyDrive/
   └── cnn-lstm/
       └── data/
           └── video_data/
               ├── fight/
               └── noFight/
   ```

4. Run the notebook cells in order to preprocess the videos, train or load the model, and perform activity prediction.

### Weapon Detection

1. Open `weapon_detection_final.ipynb` in Google Colab.
2. Enable a GPU runtime.
3. Run the YOLOv7 setup and dependency-installation cells.
4. Connect the notebook to a Roboflow project containing a YOLOv7-formatted dataset.
5. Train the custom detector and run inference on images or video input.

The Roboflow configuration follows this pattern:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov7")
```

Keep API keys and private dataset credentials out of source control.

## Technical Contribution

The project combines two complementary forms of video understanding. YOLOv7 provides frame-level weapon localization, while the CNN-LSTM model uses sequences of frames to recognize activity patterns that cannot be reliably interpreted from a single image.

The associated evaluation also examined parallel processing for multiple video feeds and image-enhancement techniques for low-light scenes. Together, these approaches improved processing throughput and the quality of visual features available to the models.

The performance figures listed above summarize the project’s reported evaluation results.

## Research Recognition

The work was presented at the **Second International Conference on Next Generation Computing Systems**, held in March 2023. The paper, **“Intelligent CCTV Surveillance System,”** received the **Best Paper** award and was published in Springer’s **Communications in Computer and Information Science (CCIS)** proceedings.

**Authors:** Gokila Harini Krishna K., Srinidhi R. G., and Uma K. V.

- [Springer CCIS book series](https://link.springer.com/series/7899)
- [ICNGCS 2023 conference report](https://psgitech.ac.in/uploads/department_event_reports/1699418675_Report_InternationalconferenceICNGCS-2023.pdf)
