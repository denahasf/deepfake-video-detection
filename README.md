# 🎭 Deepfake Video Detection System

An AI-powered application that analyzes video frames and classifies videos as **REAL** or **FAKE** using deep learning.

Built with **Python, PyTorch, EfficientNet-B0, and OpenCV**.

## 🚀 Features

- 🎥 Video frame extraction using OpenCV
- 🧠 Deep learning classification using EfficientNet-B0
- 🔍 REAL / FAKE video prediction
- 📊 Confidence score output
- ⚙️ Model training with real and fake video datasets
- 🧪 Demo mode for testing without a real dataset

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Image processing and model support |
| EfficientNet-B0 | CNN-based image classification |
| OpenCV | Video frame extraction |
| Scikit-learn | Machine learning utilities |
| FaceNet-PyTorch | Face processing utilities |

## 📁 Project Structure

```text
deepfake-video-detection/
│
├── deepfake_detector.py
├── deepfake_detector_notebook.ipynb
├── deepfake_cnn.pth
├── sort_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 💻 Installation

**1. Clone the repository**

```bash
git clone https://github.com/denahasf/deepfake-video-detection.git
```

**2. Open the project folder**

```bash
cd deepfake-video-detection
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

**Run demo mode**

```bash
python deepfake_detector.py --demo
```

**Train the model**

```bash
python deepfake_detector.py --train
```

**Detect a video**

```bash
python deepfake_detector.py path/to/video.mp4
```

## 📂 Dataset Structure

For model training, organize your dataset as follows:

```text
dataset/
│
├── real/
│   ├── video1.mp4
│   └── video2.mp4
│
└── fake/
    ├── video1.mp4
    └── video2.mp4
```

**Note:** The dataset is not included in this repository because of its size. You must provide your own suitable real and fake video dataset.

## 🧠 Model Information

This project uses **EfficientNet-B0**, a convolutional neural network (CNN), with transfer learning for video-frame classification.

The system extracts frames from videos and uses the trained model to predict whether the content is real or manipulated.

**Important:** Demo mode uses synthetic data and does not demonstrate real-world detection accuracy. Predictions depend on the quality and diversity of the training dataset.

## 🎯 Project Objective

The objective of this project is to explore how artificial intelligence and deep learning can help identify manipulated video content and support digital media authenticity.

## 👨‍💻 Author

**Dinakar**

- GitHub: [@denahasf](https://github.com/denahasf)
- Project: [Deepfake Video Detection](https://github.com/denahasf/deepfake-video-detection)

---

⭐ If you find this project interesting, consider giving the repository a star!
