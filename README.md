### Step-by-Step Setup Instructions

1.  **Create a Virtual Environment**
    It is recommended to use a virtual environment to keep dependencies isolated.
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    - **Windows:** `venv\Scripts\activate`
    - **Linux/macOS:** `source venv/bin/activate`

2.  **Install Dependencies**
    Install the required libraries from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: On Linux, you may need to install `portaudio` (e.g., `sudo apt install libportaudio2`) for `sounddevice` to work.*

3.  **Download Models (Recommended for First Run)**
    To ensure all models are ready and stored locally in your project folder, run the provided download script. This ensures the app works offline and you can see the download progress.
    
    ```bash
    python download_models.py
    ```
    - **Models Folder:** A `models/` directory will be created in the project root.
    - **Whisper (STT):** Downloads to `models/whisper/` (~150MB).
    - **NLLB-200 (Translation):** Downloads to `models/nllb/` (~2.4GB).
    
    *Note: If you skip this, `main.py` will still download the models automatically into the `models/` directory on the first run, and it will now show progress.*

    *Optional:* If you want faster downloads, you can set an environment variable with your Hugging Face token:
    ```bash
    export HF_TOKEN=your_token_here
    ```

### Instructions to Run the System
1.  Navigate to the project directory.
2.  Run the orchestrator script:
    ```bash
    python main.py
    ```
3.  The GUI will appear. Click **"Start Recording"** and speak into your microphone.
4.  Subtitles will appear in the "Bangla Translation" box.

### Notes on Improving Latency & Performance
- **Model Size:**
  - `self.whisper_model_size = "tiny"` is the fastest and recommended for CPUs.
  - If you need better accuracy, use `"base"`, `"small"`, or `"medium"`.
- **Hardware Acceleration (GPU):**
  - The system automatically detects and uses an NVIDIA GPU (CUDA) if available.
  - This significantly improves translation speed and reduces lag.
- **Chunk Duration:**
  - `self.chunk_duration = 0.8` in `main.py` balances stability and speed.
  - The system now throttles processing to every 0.5s of new data to save CPU.
- **NLLB Translation:**
  - Translation is now performed more frequently (every 3+ words) to feel more "real-time".
  - Using `num_beams=1` in `translator.py` speeds up inference.
- **Accents & Mixed Language:**
  - The system uses a specific initial prompt to improve recognition of Pakistani English and Urdu mix.
