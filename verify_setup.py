import os
import sys

def verify_setup():
    project_root = os.path.dirname(os.path.abspath(__file__))
    whisper_dir = os.path.join(project_root, "models", "whisper")
    nllb_dir = os.path.join(project_root, "models", "nllb")
    
    print("-" * 50)
    print("AI MODEL VERIFICATION TOOL")
    print("-" * 50)
    
    # 1. Check Whisper
    print(f"[1] Checking Whisper (STT) at {whisper_dir}...")
    whisper_ok = False
    if os.path.exists(whisper_dir):
        # Whisper saves as a folder with model.bin usually
        for root, dirs, files in os.walk(whisper_dir):
            if "model.bin" in files:
                size = os.path.getsize(os.path.join(root, "model.bin")) / (1024 * 1024)
                print(f"✓ Found model.bin ({size:.2f} MB)")
                whisper_ok = True
                break
    
    if whisper_ok:
        print("✓ Whisper model seems correctly installed.")
    else:
        print("! Whisper model NOT found or incomplete.")

    # 2. Check NLLB
    print(f"[2] Checking NLLB (Translation) at {nllb_dir}...")
    nllb_ok = False
    total_size = 0
    incomplete = False
    if os.path.exists(nllb_dir):
        for root, dirs, files in os.walk(nllb_dir):
            for f in files:
                if f.endswith(".incomplete"):
                    incomplete = True
                total_size += os.path.getsize(os.path.join(root, f))
        
        size_gb = total_size / (1024 * 1024 * 1024)
        print(f"Total downloaded size: {size_gb:.2f} GB")
        
        if incomplete:
            print("! WARNING: Found INCOMPLETE download (.incomplete files).")
        elif size_gb > 2.0:
            print("✓ NLLB model weights found and size looks correct.")
            nllb_ok = True
    
    if not nllb_ok:
        print("! NLLB model NOT found or incomplete.")

    print("-" * 50)
    if whisper_ok and nllb_ok:
        print("VERDICT: Everything looks good! You can run 'python main.py'.")
    else:
        print("VERDICT: Models are missing or incomplete. Run 'python download_models.py'.")
    print("-" * 50)

if __name__ == "__main__":
    verify_setup()
