# Live Translator: English/Urdu to Bangla

This project provides real-time translation of speech (English, Urdu, or mixed Pakistani English/Urdu) into Bangla subtitles using Whisper and NLLB-200.

## 🚀 Setup Instructions (Step-by-Step)

Follow these steps to set up the project on your machine from scratch.

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/azharhussaincs/live_translator_english_urdu_to_bangla.git
cd live_translator_english_urdu_to_bangla
```

### 2. Create and Activate Virtual Environment
It is recommended to use a virtual environment to keep dependencies isolated.
```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Required Dependencies
Ensure you have `pip` updated and install the requirements:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Note: On Linux, you may need to install `portaudio` for audio recording: `sudo apt install libportaudio2`*

### 4. Create Models Folder and Download Models
The models are large (around 3GB total), so we download them locally to ensure offline capability and better performance.
```bash
# Run the download script
python download_models.py
```
This will:
- Create a `models/` directory.
- Download **Whisper (Tiny)** for speech-to-text (~150MB).
- Download **NLLB-200 (600M Distilled)** for translation (~2.4GB).

### 5. Run the Application
Once the models are downloaded, start the translator:
```bash
python main.py
```

## 🛠️ Usage
1. Click **"Start Recording"** in the GUI.
2. Speak clearly into your microphone (English, Urdu, or mixed).
3. The recognized text will appear in the top box.
4. The **Bangla translation** will appear in real-time in the bottom box.

## ⚙️ Configuration & Performance
- **GPU Acceleration:** The system automatically uses NVIDIA GPU (CUDA) if available.
- **Low Latency:** Optimized with Whisper `tiny` and `num_beams=1` for faster response.
- **Pakistani Style:** Specifically tuned to handle Pakistani accents and English/Urdu mixed speech (Hinglish/Urdu-English).

---
Developed for real-time live subtitles.
