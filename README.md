# AuraHands 🪄🔥  
**Real-time Magical AR Effects with MediaPipe & OpenCV**

Turn your webcam into a wizard's playground! AuraHands uses **MediaPipe** for hand & face landmark detection + custom OpenCV effects to create glowing finger trails, yellow demon eyes, and animated fireballs on your palms — all in real-time.

Perfect beginner-friendly CV project for anyone moving from traditional dev (like .NET) into fun AI/ML experiments. Built as part of my **The ML Guppy** learning journey 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-orange?logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features
- **Glowing Finger Lines** — Neon laser trails connecting matching fingertips between both hands (when detected)
- **Yellow Eyes Mode** — Turn your irises bright yellow with dark pupils (activate with peace sign ✌️ from both hands)
- **Fireball Palms** — Summon animated, pulsing fireballs above your fists (close both hands 👊👊)
- **Gesture Controls** — Fist-based mode switching & effect triggering
- **Real-time Performance** — Runs smoothly on most laptops (MediaPipe task API for hand & face landmarker)
- **Mirror Webcam View** — Natural feel with horizontal flip

## 🎮 How to Use (Gestures)
| Gesture              | Action                              | Mode/Effect                     |
|----------------------|-------------------------------------|---------------------------------|
| One fist 👊          | Toggle between Hand / Face mode     | Switch focus                    |
| Two fists 👊👊       | Activate fireballs on palms         | Fireball mode                   |
| Two peace signs ✌️✌️ (in Face mode) | Yellow eyes effect               | Demon eyes activated            |
| Press **q**          | Quit the app                        | —                               |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam

### Installation
```bash
# Clone the repo
git clone https://github.com/ml-guppy-lab/AuraHands.git
cd AuraHands

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe numpy

python auraHands.py

```
## 📸 Screenshots

### Fireball Generation
![Fireball on palms](https://github.com/ml-guppy-lab/AuraHands/raw/main/4.png)

### Yellow Eyes Masking
![Yellow demon eyes](https://github.com/ml-guppy-lab/AuraHands/raw/main/3.png)

### Face Landmarks Detection
![Face mesh detection](https://github.com/ml-guppy-lab/AuraHands/raw/main/2.png)

### Glowing Finger Lines
![Neon lines between fingers](https://github.com/ml-guppy-lab/AuraHands/raw/main/1.png)


(For the full dramatic 30-second version with Lana Del Rey "Salvatore" soundtrack and magic vibes, check out the reel on my Instagram: @TheMLGuppy)
[![TheMLGuppy](https://www.instagram.com/p/DV6E8cziUx_/)

**🛠️ How It Works (Quick Tech Breakdown)**

MediaPipe Task API (new v0.10+ style) for HandLandmarker & FaceLandmarker
Auto model download from Google Storage
Fist detection via distance thresholds (tip-to-wrist)
Custom OpenCV drawing: layered circles for fireballs + particles, gradient lines for lasers, scaled iris overlays
Gesture cooldown to avoid spam toggles

Code is kept simple & commented — perfect for learning CV basics!

**🔮 Future Ideas / Improvements**

Smoother tracking (Kalman filter? interpolation?)
More gestures (thumbs up → confetti, etc.)
Add sound effects / voice commands
Mobile version (perhaps with MediaPipe Python → Flutter/MAUI bridge)
Custom effects via config file
