"""
Deepfake Detection Model
========================
Uses OpenCV for frame extraction and a CNN (EfficientNet-B0) for classification.
Output: REAL or FAKE with confidence score.

Requirements:
    pip install torch torchvision opencv-python facenet-pytorch scikit-learn tqdm Pillow

Dataset structure expected:
    dataset/
        real/   ← .mp4 / .avi / .mov files of genuine videos
        fake/   ← .mp4 / .avi / .mov files of deepfake videos
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DATASET_DIR  = "dataset"          # root folder with real/ and fake/ subfolders
MODEL_PATH   = "deepfake_cnn.pth"
FRAME_COUNT  = 20                 # frames to sample per video
IMG_SIZE     = 224                # EfficientNet input size
BATCH_SIZE   = 8
EPOCHS       = 20
LR           = 1e-4
VAL_SPLIT    = 0.2
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 2. FRAME EXTRACTION (OpenCV)
# ─────────────────────────────────────────────
def extract_frames(video_path: str, num_frames: int = FRAME_COUNT) -> list[np.ndarray]:
    """
    Extract `num_frames` evenly-spaced frames from a video using OpenCV.
    Returns a list of RGB numpy arrays (H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    # Pick evenly-spaced frame indices
    indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)

    cap.release()
    return frames


def preprocess_frame(frame: np.ndarray, size: int = IMG_SIZE) -> torch.Tensor:
    """
    Convert a raw RGB frame to a normalised tensor ready for EfficientNet.
    Applies: resize → center-crop → normalise.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                             std =[0.229, 0.224, 0.225]),
    ])
    img = Image.fromarray(frame)
    return transform(img)  # shape: (3, size, size)


# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

class DeepfakeVideoDataset(Dataset):
    """
    Loads videos from:
        dataset/real/  → label 0
        dataset/fake/  → label 1

    Each item is the mean of all extracted frame tensors for that video.
    This gives a single (3, 224, 224) tensor per video, suitable for a CNN.
    """

    def __init__(self, dataset_dir: str, num_frames: int = FRAME_COUNT):
        self.samples: list[tuple[torch.Tensor, int]] = []
        self.num_frames = num_frames

        for label, subfolder in enumerate(["real", "fake"]):
            folder = os.path.join(dataset_dir, subfolder)
            if not os.path.isdir(folder):
                print(f"  [!] Missing folder: {folder}")
                continue

            video_files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
            ]
            print(f"  Found {len(video_files)} videos in '{subfolder}/'")

            for vpath in tqdm(video_files, desc=f"Loading {subfolder}", leave=False):
                frames = extract_frames(vpath, num_frames)
                if not frames:
                    continue
                tensors = [preprocess_frame(f) for f in frames]
                # Average across frames → single representative tensor
                mean_tensor = torch.stack(tensors).mean(dim=0)
                self.samples.append((mean_tensor, label))

        print(f"\nDataset ready: {len(self.samples)} samples "
              f"({sum(1 for _,l in self.samples if l==0)} real, "
              f"{sum(1 for _,l in self.samples if l==1)} fake)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return x, torch.tensor(y, dtype=torch.float32)


# ─────────────────────────────────────────────
# 4. CNN MODEL (EfficientNet-B0 + custom head)
# ─────────────────────────────────────────────
def build_model() -> nn.Module:
    """
    EfficientNet-B0 with a binary classification head.
    Early layers frozen for transfer learning — only the last block
    and the new head are trained. This works well on small datasets.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last feature block (features[7] and features[8])
    for layer in [model.features[7], model.features[8]]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace the classifier head
    in_features = model.classifier[1].in_features  # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),          # output ∈ (0, 1): probability of FAKE
    )

    return model


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
def train_model(dataset_dir: str = DATASET_DIR) -> nn.Module:
    """Full training loop with validation. Saves best checkpoint."""
    print("\n─── Loading dataset ───")
    dataset = DeepfakeVideoDataset(dataset_dir)

    if len(dataset) < 4:
        raise ValueError("Need at least 4 videos (2 real + 2 fake) to train.")

    val_size   = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n─── Building model ───")
    model   = build_model().to(DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    print(f"\n─── Training for {EPOCHS} epochs ───")

    for epoch in range(1, EPOCHS + 1):
        # ── Train phase ──
        model.train()
        train_loss, train_correct = 0.0, 0

        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [train]", leave=False):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(frames).squeeze(1)
            loss  = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(labels)
            train_correct += ((preds > 0.5).float() == labels).sum().item()

        # ── Validation phase ──
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(DEVICE), labels.to(DEVICE)
                preds = model(frames).squeeze(1)
                loss  = loss_fn(preds, labels)
                val_loss    += loss.item() * len(labels)
                val_correct += ((preds > 0.5).float() == labels).sum().item()
                all_preds.extend((preds > 0.5).float().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        t_loss = train_loss / train_size
        v_loss = val_loss   / val_size
        t_acc  = train_correct / train_size
        v_acc  = val_correct   / val_size

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train loss: {t_loss:.4f}, acc: {t_acc:.1%} | "
              f"Val loss: {v_loss:.4f}, acc: {v_acc:.1%}")

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model → {MODEL_PATH}")

    # Final validation report
    print("\n─── Validation report (best model) ───")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in val_loader:
            preds = model(frames.to(DEVICE)).squeeze(1)
            all_preds.extend((preds > 0.5).cpu().numpy().astype(int))
            all_labels.extend(labels.numpy().astype(int))

    print(classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"]))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return model


# ─────────────────────────────────────────────
# 6. INFERENCE
# ─────────────────────────────────────────────
def predict(video_path: str, model_path: str = MODEL_PATH) -> dict:
    """
    Predict whether a single video is REAL or FAKE.

    Returns:
        {
            "label":       "REAL" or "FAKE",
            "confidence":  float (0–100),
            "fake_prob":   float (0.0–1.0),
            "frames_used": int,
        }
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found at '{model_path}'. "
            "Run train_model() first."
        )

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    frames = extract_frames(video_path)
    if not frames:
        return {"label": "ERROR", "confidence": 0.0,
                "fake_prob": 0.0, "frames_used": 0,
                "message": "No frames could be extracted from the video."}

    tensors = [preprocess_frame(f) for f in frames]
    mean_tensor = torch.stack(tensors).mean(dim=0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_prob = model(mean_tensor).item()

    label      = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if fake_prob > 0.5 else 1.0 - fake_prob

    return {
        "label":       label,
        "confidence":  round(confidence * 100, 1),
        "fake_prob":   round(fake_prob, 4),
        "frames_used": len(frames),
    }


def predict_batch(video_paths: list[str], model_path: str = MODEL_PATH) -> list[dict]:
    """Run predict() on a list of videos and print a summary table."""
    results = []
    for path in video_paths:
        print(f"Analysing: {os.path.basename(path)}")
        result = predict(path, model_path)
        result["video"] = os.path.basename(path)
        results.append(result)

    # Print summary table
    print("\n" + "─" * 60)
    print(f"{'Video':<30} {'Result':<8} {'Confidence':>10}")
    print("─" * 60)
    for r in results:
        name = r["video"][:28]
        print(f"{name:<30} {r['label']:<8} {r['confidence']:>9.1f}%")
    print("─" * 60)

    return results


# ─────────────────────────────────────────────
# 7. DEMO RUNNER (no dataset needed)
# ─────────────────────────────────────────────
def run_demo_with_synthetic_data():
    """
    Generates tiny synthetic 'videos' (solid-colour MP4 files) so you
    can verify the entire pipeline works before you have real data.
    This is only for testing — accuracy will be meaningless on synthetic data.
    """
    import tempfile

    print("\n─── Running demo with synthetic data ───")
    demo_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(demo_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "fake"), exist_ok=True)

    def make_video(path, color_bgr, num_frames=60):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30, (112, 112))
        frame  = np.full((112, 112, 3), color_bgr, dtype=np.uint8)
        for _ in range(num_frames):
            writer.write(frame)
        writer.release()

    # Create 4 real + 4 fake synthetic videos
    for i in range(4):
        make_video(os.path.join(demo_dir, "real", f"real_{i}.mp4"), (0, 120, 200))
        make_video(os.path.join(demo_dir, "fake", f"fake_{i}.mp4"), (200, 50,   0))

    print(f"Synthetic dataset written to: {demo_dir}")
    print("(Colour-only videos — accuracy numbers are not meaningful here.)\n")

    model = train_model(demo_dir)

    # Test prediction on one of the synthetic videos
    test_video = os.path.join(demo_dir, "real", "real_0.mp4")
    result = predict(test_video)
    print(f"\nPrediction for '{os.path.basename(test_video)}':")
    print(f"  → {result['label']} ({result['confidence']}% confidence)")
    print(f"     fake_prob={result['fake_prob']}, frames_used={result['frames_used']}")

    return model


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "--demo":
        # Quick sanity-check with synthetic data
        run_demo_with_synthetic_data()

    elif len(sys.argv) == 2 and sys.argv[1].endswith((".mp4", ".avi", ".mov", ".mkv")):
        # Predict on a single video
        video_path = sys.argv[1]
        print(f"\nAnalysing: {video_path}")
        result = predict(video_path)
        print(f"\n{'─'*40}")
        print(f"  Result     : {result['label']}")
        print(f"  Confidence : {result['confidence']}%")
        print(f"  Fake prob  : {result['fake_prob']}")
        print(f"  Frames used: {result['frames_used']}")
        print(f"{'─'*40}")

    elif len(sys.argv) == 2 and sys.argv[1] == "--train":
        # Train on your own dataset
        if not os.path.isdir(DATASET_DIR):
            print(f"[ERROR] Dataset folder '{DATASET_DIR}' not found.")
            print("Create it with real/ and fake/ subfolders containing .mp4 files.")
            sys.exit(1)
        train_model(DATASET_DIR)

    else:
        print(__doc__)
        print("Usage:")
        print("  python deepfake_detector.py --train          # train on dataset/")
        print("  python deepfake_detector.py video.mp4        # predict one video")
        print("  python deepfake_detector.py --demo           # run synthetic demo")
