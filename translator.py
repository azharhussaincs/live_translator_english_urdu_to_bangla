import os
import sys

# Setting HF Hub environment variables BEFORE any library imports
# "error" verbosity silences most warnings, but we'll enable info later during download
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device="cpu"):
        """
        Initializes the NLLB-200 translation model.
        The 600M distilled version is relatively lightweight for CPUs.
        """
        project_root = os.path.dirname(os.path.abspath(__file__))
        nllb_dir = os.path.join(project_root, "models", "nllb")
        os.makedirs(nllb_dir, exist_ok=True)
        
        # Use provided token if available, otherwise just warn the user.
        hf_token = os.getenv("HF_TOKEN")
        
        # Explicitly set token=None if no token is found to avoid ambiguity.
        token_arg = hf_token if hf_token else False
        
        print(f"Loading translation model: {model_name} on {device}...")
        self.device = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
        
        # First attempt: load with local_files_only=True to prioritize cached models
        # and avoid checking the Hub (this is faster and more reliable if pre-downloaded).
        try:
            print(f"Checking for model in local directory: {nllb_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=nllb_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True, cache_dir=nllb_dir)
            print("Translation model loaded from local directory.")
        except Exception:
            print("Model not found locally. Attempting to download from HF Hub...")
            # Re-enable info logging temporarily to see download progress
            from huggingface_hub import logging as hf_logging
            from transformers.utils import logging as tf_logging
            hf_logging.set_verbosity_info()
            tf_logging.set_verbosity_info()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token_arg, cache_dir=nllb_dir)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token_arg, cache_dir=nllb_dir)
                print("Translation model downloaded and loaded successfully.")
            except Exception as e:
                print(f"Critical Error: Failed to load translation model: {e}")
                print("Initial runs require an internet connection to download models.")
                print("Alternatively, run 'python download_models.py' first.")
                raise
            finally:
                # Silence it again after download
                hf_logging.set_verbosity_error()
                tf_logging.set_verbosity_error()

        # Initialize the model and tokenizer are loaded.
        # Direct model calls are now used in translate() to avoid pipeline issues.
        pass

    def translate(self, text, src_lang=None):
        """
        Translates text to Bangla (bn).
        NLLB codes:
        - Urdu: urd_Arab
        - English: eng_Latn
        - Bangla: ben_Beng
        """
        if not text:
            return ""
        
        # If Whisper detects English, we can adjust src_lang dynamically
        # for better performance.
        if src_lang == "en":
            src = "eng_Latn"
        elif src_lang == "ur":
            src = "urd_Arab"
        elif src_lang == "hi":
            src = "hin_Deva"  # Hindi is very similar to Urdu spoken, sometimes Whisper picks 'hi'
        else:
            # For mixed speech, eng_Latn or urd_Arab are best. 
            # If nothing detected, we default to a mix-friendly code or Urdu.
            src = "urd_Arab"

        try:
            # 1. Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            # 2. Generate translated IDs. NLLB uses the target language code as 
            # the beginning of the forced BOS token.
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids("ben_Beng")
            
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=1,
                do_sample=False
            )
            
            # 3. Decode back to text
            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original if translation fails
