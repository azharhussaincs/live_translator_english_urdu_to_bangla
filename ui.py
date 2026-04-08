import tkinter as tk
from tkinter import scrolledtext
import threading

class LiveSubtitleUI:
    def __init__(self, title="Live Bangla Subtitles", on_start=None, on_stop=None):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e1e")

        # Top Section: Control Buttons
        self.control_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.control_frame.pack(pady=10)

        self.start_btn = tk.Button(self.control_frame, text="Start Recording", command=on_start, 
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(self.control_frame, text="Stop Recording", command=on_stop,
                                  bg="#f44336", fg="white", font=("Arial", 12, "bold"), state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Labels
        self.label_orig = tk.Label(self.root, text="Recognized Speech (Mixed English/Urdu):", 
                                   fg="#888888", bg="#1e1e1e", font=("Arial", 10))
        self.label_orig.pack(anchor="w", padx=20)

        # Recognized text display
        self.orig_text = scrolledtext.ScrolledText(self.root, height=5, font=("Arial", 14), bg="#2d2d2d", fg="#ffffff", state=tk.DISABLED)
        self.orig_text.pack(fill=tk.X, padx=20, pady=5)

        self.label_trans = tk.Label(self.root, text="Bangla Translation (বাংলা):", 
                                    fg="#888888", bg="#1e1e1e", font=("Arial", 10))
        self.label_trans.pack(anchor="w", padx=20)

        # Bangla text display
        self.trans_text = scrolledtext.ScrolledText(self.root, height=10, font=("Arial", 18, "bold"), bg="#2d2d2d", fg="#4CAF50", state=tk.DISABLED)
        self.trans_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

    def update_original(self, text):
        """Updates the recognized text area."""
        self.orig_text.config(state=tk.NORMAL)
        self.orig_text.delete(1.0, tk.END)
        self.orig_text.insert(tk.END, text)
        self.orig_text.see(tk.END) # Auto-scroll to bottom
        self.orig_text.config(state=tk.DISABLED)

    def update_translation(self, text):
        """Updates the translated text area."""
        self.trans_text.config(state=tk.NORMAL)
        self.trans_text.delete(1.0, tk.END)
        self.trans_text.insert(tk.END, text)
        self.trans_text.see(tk.END) # Auto-scroll to bottom
        self.trans_text.config(state=tk.DISABLED)

    def set_status_running(self):
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def set_status_stopped(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def run(self):
        self.root.mainloop()
