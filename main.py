import threading
import time
import sys
import os
import numpy as np
from audio_stream import AudioStream
from speech_to_text import SpeechToText
from translator import Translator
from ui import LiveSubtitleUI

class Orchestrator:
    def __init__(self):
        # Configuration
        self.sample_rate = 16000
        self.whisper_model_size = "tiny"  # Use 'tiny' or 'base' for low latency
        self.chunk_duration = 0.8        # Decreased for faster word-by-word feel
        self.max_buffer_duration = 10.0  # Allow longer sentences
        
        # Performance/Latency: Use GPU if available
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Language Optimization: Specifically for Urdu/English mix (Pakistani English style)
        self.forced_language = None      # 'ur' or 'en' if you want to force it
        # Prompt should be words, not sentences, to avoid hallucinations repeating them
        # Removed specific greetings to avoid hallucinations when silent
        self.initial_prompt = "English, Urdu, Pakistani style, Hinglish, translation, lecture, speech."
        
        # Inform user about initial setup
        print("-" * 50)
        print("Starting Real-Time Translation System...")
        print("Priority: Loading models from local project directory ('models/').")
        print("-" * 50)
        
        # Initialize components
        try:
            self.stt = SpeechToText(model_size=self.whisper_model_size, device=self.device, compute_type=self.compute_type)
            self.translator = Translator(device=self.device)
        except Exception as e:
            print(f"\nInitialization failed: {e}")
            sys.exit(1)
            
        self.audio = AudioStream(sample_rate=self.sample_rate)
        
        # UI is initialized but not yet started
        self.ui = LiveSubtitleUI(
            on_start=self.start_processing, 
            on_stop=self.stop_processing
        )
        
        self.is_running = False
        self.process_thread = None
        
        # History buffers to keep UI content across chunks
        self.transcription_history = []
        self.translation_history = []
        self.max_history_lines = 10

    def start_processing(self):
        """Callback to start audio streaming and processing thread."""
        if not self.is_running:
            self.is_running = True
            self.ui.set_status_running()
            self.audio.start()
            self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.process_thread.start()
            print("Processing loop started.")

    def stop_processing(self):
        """Callback to stop audio streaming and processing thread."""
        if self.is_running:
            self.is_running = False
            self.audio.stop()
            self.ui.set_status_stopped()
            print("Processing loop stopped.")

    def _process_loop(self):
        """Main loop that collects audio and performs STT + translation."""
        audio_buffer = []
        chunks_per_second = int(self.sample_rate / 1024)
        chunks_per_process = int(self.chunk_duration * chunks_per_second)
        max_chunks = int(self.max_buffer_duration * chunks_per_second)
        
        last_text = ""
        last_process_time = time.time()
        
        while self.is_running:
            # Collect audio chunks from the queue
            chunk = self.audio.get_audio_chunk()
            if chunk is not None:
                audio_buffer.append(chunk)
            
            # Keep buffer size limited
            if len(audio_buffer) > max_chunks:
                audio_buffer = audio_buffer[-max_chunks:]
            
            # Process as soon as we have enough audio for a chunk AND a reasonable time has passed
            current_time = time.time()
            if len(audio_buffer) >= chunks_per_process and (current_time - last_process_time) >= 0.5:
                last_process_time = current_time
                audio_data = np.concatenate(audio_buffer, axis=0).flatten()
                
                # Pre-check for silence to avoid sending empty/noisy audio to Whisper
                rms = np.sqrt(np.mean(audio_data**2))
                if rms < 0.008:  # Slightly increased from 0.005 to filter very faint noise
                    if len(audio_buffer) > chunks_per_second * 2: # Clear if silent for > 2s
                         audio_buffer = []
                    time.sleep(0.1)
                    continue

                # Perform Speech-To-Text with language bias
                text, detected_lang = self.stt.transcribe(
                    audio_data, 
                    language=self.forced_language,
                    initial_prompt=self.initial_prompt
                )
                
                if text and text != last_text:
                    # Update UI with current history + live text
                    history_str = "\n".join(self.transcription_history)
                    if history_str:
                        display_text = f"{history_str}\n[{detected_lang}] {text}"
                    else:
                        display_text = f"[{detected_lang}] {text}"
                    self.ui.update_original(display_text)
                    
                    # Perform Translation to Bangla
                    # Optimization: Only translate when text seems stable or finished
                    is_finished = any(text.rstrip().endswith(p) for p in [".", "?", "!", "।", "۔"])
                    
                    if is_finished or len(text.split()) >= 3: # Reduced from 5 to 3 for more real-time feel
                        bangla_text = self.translator.translate(text, src_lang=detected_lang)
                        
                        history_trans_str = "\n".join(self.translation_history)
                        if history_trans_str:
                            display_trans = f"{history_trans_str}\n{bangla_text}"
                        else:
                            display_trans = bangla_text
                        self.ui.update_translation(display_trans)
                    
                    # If the text seems finished, move to history
                    if is_finished:
                        # Re-translate final version for accuracy
                        bangla_text = self.translator.translate(text, src_lang=detected_lang)
                        self.transcription_history.append(f"[{detected_lang}] {text}")
                        self.translation_history.append(bangla_text)
                        
                        # Update UI one last time for this sentence
                        history_str = "\n".join(self.transcription_history)
                        self.ui.update_original(history_str)
                        history_trans_str = "\n".join(self.translation_history)
                        self.ui.update_translation(history_trans_str)

                        # Limit history
                        if len(self.transcription_history) > self.max_history_lines:
                            self.transcription_history.pop(0)
                            self.translation_history.pop(0)
                            
                        # Clear audio buffer to start fresh for next sentence
                        audio_buffer = []
                        last_text = ""
                    else:
                        last_text = text
                
                # Small delay to prevent CPU spinning if chunk is very small
                time.sleep(0.1)
            
            time.sleep(0.01)

    def run(self):
        """Starts the main UI loop."""
        print("Application ready. Use UI buttons to start/stop.")
        self.ui.run()

if __name__ == "__main__":
    app = Orchestrator()
    app.run()
