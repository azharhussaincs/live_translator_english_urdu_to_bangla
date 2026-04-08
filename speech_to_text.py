import os
import sys

# Setting HF Hub environment variables BEFORE any library imports
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8"):
        project_root = os.path.dirname(os.path.abspath(__file__))
        whisper_dir = os.path.join(project_root, "models", "whisper")
        os.makedirs(whisper_dir, exist_ok=True)
            
        print(f"Loading Whisper model: {model_size} on {device}...")
        try:
            # First attempt to load with local_files_only=True to prioritize local models/ directory.
            # We use the model_size string which faster-whisper will check inside download_root.
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type, 
                                     download_root=whisper_dir, local_files_only=True)
            print(f"Whisper model loaded from local directory: {whisper_dir}")
        except Exception:
            print(f"Whisper model '{model_size}' not found in local project directory. Attempting to download...")
            # Re-enable info logging temporarily to see download progress if supported
            from huggingface_hub import logging as hf_logging
            hf_logging.set_verbosity_info()
            try:
                # To suppress the "unauthenticated requests" warning, we use the download_model 
                # utility from faster_whisper which allows passing an explicit HF token (or False).
                from faster_whisper.utils import download_model
                
                # download_model returns the path to the downloaded model
                model_path = download_model(
                    model_size,
                    output_dir=whisper_dir,
                    local_files_only=False,
                    use_auth_token=False  # Explicitly disable token to silence warning
                )
                
                # Now load from the explicitly downloaded path
                self.model = WhisperModel(model_path, device=device, compute_type=compute_type, 
                                         local_files_only=True)
                print("Whisper model downloaded and loaded successfully.")
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
                raise
            finally:
                hf_logging.set_verbosity_error()

    def transcribe(self, audio_data, language=None, initial_prompt=None):
        """
        Transcribes audio data (numpy array).
        Returns a string of recognized text and detected language.
        """
        # For mixed English/Urdu, providing an initial prompt helps Whisper
        # understand the context and expected languages better.
        prompt = initial_prompt or "English, Urdu, lecture, speech."
        
        segments, info = self.model.transcribe(
            audio_data, 
            beam_size=1, # Lower beam size for faster real-time processing
            language=language,
            initial_prompt=prompt,
            vad_filter=True,
            no_speech_threshold=0.5, # Reduced from 0.6 to be more sensitive
            log_prob_threshold=-1.0, 
            vad_parameters=dict(
                min_silence_duration_ms=600, # Reduced for faster response
                threshold=0.4, # Lowered from 0.5 to be more sensitive to quiet voice
                min_speech_duration_ms=100 # Reduced from 150 to catch short words
            )
        )
        text = " ".join([segment.text for segment in segments])
        return text.strip(), info.language
