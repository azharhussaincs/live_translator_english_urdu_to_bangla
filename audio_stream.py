import sounddevice as sd
import queue

class AudioStream:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.stream = None

    def _audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def start(self):
        """Starts the audio recording stream."""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()
        print("Audio stream started...")

    def stop(self):
        """Stops the audio recording stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Audio stream stopped.")

    def get_audio_chunk(self):
        """Returns the oldest audio chunk from the queue, if any."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
