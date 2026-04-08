import os
import sys

# Silencing HF Hub warning before any Hugging Face imports
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

def download_models():
    # Define model storage directory within the project
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    whisper_dir = os.path.join(models_dir, "whisper")
    nllb_dir = os.path.join(models_dir, "nllb")
    
    print("-" * 50)
    print("Pre-downloading models to local project directory...")
    print(f"Models will be stored in: {models_dir}")
    print("-" * 50)

    # Re-enable info logging to see progress bars
    from huggingface_hub import logging as hf_logging
    from transformers.utils import logging as tf_logging
    hf_logging.set_verbosity_info()
    tf_logging.set_verbosity_info()

    try:
        from faster_whisper.utils import download_model
        print("\n[1/2] Downloading Whisper (STT) model: tiny (~150MB)...")
        # Explicitly passing use_auth_token=False to suppress unauthenticated requests warning
        download_model("tiny", output_dir=whisper_dir, local_files_only=False, use_auth_token=False)
        print(f"Whisper model downloaded and cached in {whisper_dir}")
    except Exception as e:
        print(f"Error downloading Whisper model: {e}")

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_name = "facebook/nllb-200-distilled-600M"
        print(f"\n[2/2] Downloading Translation (NLLB) model: {model_name} (~2.4GB)...")
        print("Progress will be shown below (percentage and speed):")
        
        # Explicitly passing token=False to avoid authentication warnings
        AutoTokenizer.from_pretrained(model_name, token=False, cache_dir=nllb_dir)
        AutoModelForSeq2SeqLM.from_pretrained(model_name, token=False, cache_dir=nllb_dir)
        print(f"Translation model downloaded and cached in {nllb_dir}")
    except Exception as e:
        print(f"Error downloading Translation model: {e}")

    # Silence logging again
    hf_logging.set_verbosity_error()
    tf_logging.set_verbosity_error()

    print("\n" + "-" * 50)
    print("All models are now downloaded and cached locally.")
    print("You can now run 'python main.py' to start the application.")
    print("-" * 50)

if __name__ == "__main__":
    download_models()
