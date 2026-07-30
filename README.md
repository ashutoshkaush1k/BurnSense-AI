# 🩹 BurnSense AI

**A dual-stage deep learning diagnostic assistant for classifying burn severity.**

![BurnSense UI](link)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-U--Net%20%2B%20ResNet18-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

> ⚠️ **Disclaimer:** BurnSense AI is a triage-guidance research project, **not** a substitute for professional medical care. If you or someone else has a serious burn, seek real medical attention immediately.

---

## 🤔 The "Why" — Explained Simply

Diagnosing a burn from a photo is really two separate questions, and most models try to answer both at once with a single blurry guess. BurnSense AI splits the job into two specialists that do one thing extremely well:

- 👀 **The Eyes (U-Net):** A segmentation model whose *only* job is to stare at the photo and answer "where, exactly, is the burn?" It draws a precise pixel-level boundary around the injury — nothing more, nothing less.
- 🧠 **The Brain (ResNet18):** Once the eyes have found the burn, the brain's job is to look *specifically* at that region and reason about it: how deep does this go, how severe is it, is this 1st, 2nd, or 3rd degree?

This is the same division of labor a real clinician uses — first you *locate* the injury, then you *assess* it. By training one model per task instead of forcing one network to do both, each half gets to be really good at its narrow job, and the mistakes of "I detected something vaguely burn-shaped, so I'll just guess a random severity" become far less likely.

---

## ✨ Core Innovations

### 1. Smart Context Padding
A raw segmentation mask is often *too* precise — it crops out exactly the burn and nothing else. But real diagnosis needs context: how does the burn tissue compare to the healthy skin right next to it? BurnSense AI takes the U-Net's bounding box and **pads it out by 20%**, pulling in a ring of surrounding healthy tissue before handing the crop to the ResNet18 classifier. This extra context measurably reduces false positives and low-confidence guesses compared to feeding the classifier a tightly-cropped, context-free mask.

### 2. Invalid Image Safety Net
Every medical AI demo eventually gets shown a photo of a car, a pizza, or a cat — and a naive model will confidently tell you it's a "2nd-degree burn." BurnSense AI's FastAPI backend actively guards against this: if the segmented burn area falls below a minimum size **or** the classifier's confidence falls below a minimum threshold, the API refuses to return a diagnosis and instead flags the image as invalid. This keeps the tool clinically honest instead of hallucinating an answer for images it was never meant to analyze.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python · PyTorch (custom U-Net + ResNet18 transfer learning) · FastAPI · Uvicorn |
| **Frontend** | Vanilla HTML5 · CSS3 (Glassmorphism, custom dynamic cursor-tracking spotlight) · JavaScript |

---

## 🚀 Local Setup / Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/burnsense-ai.git
cd burnsense-ai
```

**2. Install the Python dependencies**
```bash
pip install torch torchvision fastapi uvicorn python-multipart pillow numpy tqdm matplotlib opencv-python
```
> 💡 If you have an NVIDIA GPU and want CUDA acceleration, install the matching `torch`/`torchvision` build from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the default CPU wheels above.

**3. Add your trained model weights**
Place your trained checkpoints in `checkpoints/`:
- `checkpoints/debiased_model.pth` — the U-Net segmentation model
- `checkpoints/burn_degree_classifier.pth` — the ResNet18 degree classifier

**4. Start the backend**

Double-click `run_app.bat` (or run it from a terminal):
```bash
run_app.bat
```
This launches the FastAPI backend at `http://localhost:8000`.

**5. Open the frontend**

Open `frontend/index.html` directly in your browser, or serve it locally:
```bash
python -m http.server 8420 --directory frontend
```
Then visit `http://localhost:8420` — upload a burn photo and watch the full pipeline run end-to-end.

---

## 👤 Author & Credits

Built and maintained by **Ashutosh Kaushik**
*MBA Tech in Computer Engineering, NMIMS*
