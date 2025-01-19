import logging
import sys
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread, Lock
from typing import Optional

import cv2
import mss
import numpy as np
import pyttsx3
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GPUStats:

    memory_allocated: float
    memory_total: float
    memory_percent: float

    def __str__(self) -> str:
        return (
            f"GPU Memory Usage: {self.memory_percent:.2f}% | "
            f"Allocated: {self.memory_allocated:.2f} MB / {self.memory_total:.2f} MB"
        )


class TTSEngine:
    def __init__(self, rate: int = 150):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.queue = Queue()
        self.lock = Lock()
        self.speaking = False
        self.running = True
        self.thread = Thread(target=self._speaker_worker, daemon=True)
        self.thread.start()

    def _speaker_worker(self):
        while self.running:
            if not self.queue.empty() and not self.speaking:
                text = self.queue.get()
                with self.lock:
                    self.speaking = True
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        logger.error(f"TTS error: {str(e)}")
                    finally:
                        self.speaking = False
            time.sleep(0.1)

    def say(self, text: str):
        if self.queue.qsize() < 2:  # Limit queue size to prevent backup
            self.queue.put(text)

    def stop(self):
        self.running = False
        with self.lock:
            try:
                self.engine.stop()
            except Exception as e:
                logger.error(f"Error stopping TTS: {str(e)}")
        self.thread.join(timeout=1.0)


class GPUMonitor:
    @staticmethod
    def get_gpu_stats() -> Optional[GPUStats]:
        if not torch.cuda.is_available():
            return None

        memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (
            1024**2
        )
        memory_percent = (memory_allocated / memory_total) * 100

        return GPUStats(memory_allocated, memory_total, memory_percent)


class CaptionGenerator:
    def __init__(
        self,
        processor: AutoProcessor,
        model: AutoModelForImageTextToText,
        device: str,
        speech_rate: int = 150,
    ):
        self.processor = processor
        self.model = model
        self.device = device
        self.current_caption = f"Initializing caption... ({device.upper()})"
        self.last_logged_caption = ""
        self.caption_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.running = True
        self.tts_engine = TTSEngine(speech_rate)
        self.thread = Thread(target=self._caption_worker, daemon=True)
        self.thread.start()

    def _caption_worker(self):
        while self.running:
            try:
                if not self.caption_queue.empty():
                    frame = self.caption_queue.get()
                    caption = self._generate_caption(frame)
                    with self.lock:
                        if caption != self.current_caption:
                            self.current_caption = caption
                            self.tts_engine.say(caption)
            except Exception as e:
                logger.error(f"Caption worker error: {str(e)}")
            time.sleep(0.1)

    def _generate_caption(self, image: np.ndarray) -> str:
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {
                name: tensor.to(self.device) for name, tensor in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=5,
                    num_return_sequences=1,
                )
            caption = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0].strip()
            return caption
        except Exception as e:
            logger.error(f"Caption generation error: {str(e)}")
            return f"Caption generation failed ({self.device.upper()})"

    def update_frame(self, frame: np.ndarray):
        if self.caption_queue.empty():
            try:
                self.caption_queue.put_nowait(frame.copy())
            except Exception as e:
                logger.error(f"Failed to update frame: {str(e)}")

    def get_caption(self) -> str:
        with self.lock:
            return self.current_caption

    def stop(self):
        self.running = False
        self.tts_engine.stop()
        self.thread.join(timeout=1.0)


class ScreenCaptioner:
    """Main class for screen capture and caption generation."""

    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cpu"

    def load_models(self) -> bool:
        try:
            self.processor = AutoProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.device == "cuda":
                torch.cuda.set_per_process_memory_fraction(0.9)
                self.model = self.model.to("cuda")

            return True
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return False

    def run(self):
        logger.info(f"Screen capture started using {self.device.upper()}.")
        caption_generator = CaptionGenerator(
            self.processor, self.model, self.device
        )
        prev_time = time.time()

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            try:
                while True:
                    screen = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                    caption_generator.update_frame(frame)

                    current_caption = caption_generator.get_caption()
                    gpu_stats = GPUMonitor.get_gpu_stats()
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time

                    if (
                        current_caption
                        != caption_generator.last_logged_caption
                    ):
                        gpu_info = (
                            str(gpu_stats)
                            if gpu_stats
                            else "GPU not available"
                        )
                        logger.info(
                            f"Caption: {current_caption} | {gpu_info} | FPS: {fps:.2f}"
                        )
                        caption_generator.last_logged_caption = current_caption

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            except KeyboardInterrupt:
                logger.info("Screen capture interrupted by user.")
            finally:
                caption_generator.stop()
                cv2.destroyAllWindows()


def main():
    logger.info("Initializing Screen Caption application...")

    captioner = ScreenCaptioner()
    if not captioner.load_models():
        logger.error("Failed to initialize. Exiting.")
        sys.exit(1)

    logger.info(f"Using {captioner.device.upper()} for inference.")
    logger.info("Starting live screen capture with BLIP captioning and TTS...")

    captioner.run()


if __name__ == "__main__":
    main()
