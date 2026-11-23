# Deepfake Video Detection System

An AI-powered video deepfake detection system using **EfficientNet-B0, PyTorch, and OpenCV** to classify videos as REAL or FAKE.

## Features
- Video frame extraction using OpenCV
- Deep learning classification using EfficientNet-B0
- REAL / FAKE prediction
- Confidence score output
- Model training using real and fake video datasets
- Demo mode for testing without a real dataset

## Technologies Used
- Python
- PyTorch
- Torchvision
- OpenCV
- EfficientNet-B0
- Scikit-learn
- Facenet-PyTorch

## Project Structure

\\\
deepfake-video-detection/
+-- deepfake_detector.py
+-- deepfake_detector_notebook.ipynb
+-- deepfake_cnn.pth
+-- sort_dataset.py
+-- requirements.txt
+-- .gitignore
+-- README.md
\\\

## Installation

Clone the repository:

\\\ash
git clone https://github.com/denahasf/deepfake-video-detection.git
cd deepfake-video-detection
\\\

Install dependencies:

\\\ash
pip install -r requirements.txt
\\\

## Usage

Run demo:

\\\ash
python deepfake_detector.py --demo
\\\

Train the model:

\\\ash
python deepfake_detector.py --train
\\\

Predict a video:

\\\ash
python deepfake_detector.py path/to/video.mp4
\\\

## Dataset

The training script expects this folder structure:

\\\
dataset/
+-- real/
+-- fake/
\\\

Place genuine videos in eal/ and manipulated videos in ake/.

The dataset is not included in this repository because of its large size. Obtain and organize a suitable dataset locally before training.

## Model

The project uses EfficientNet-B0 with transfer learning for video-frame classification.

**Note:** Predictions depend on the quality and diversity of the training data. Demo mode uses synthetic data and does not demonstrate real-world detection accuracy.

## Author

**Dinakar**

GitHub: [@denahasf](https://github.com/denahasf)
