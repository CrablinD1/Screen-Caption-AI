# Screen Caption AI

Live screen captioning using AI. This tool watches your screen and tells you what it sees in real-time, with voice output.

## What it does

- Captures your screen in real-time
- Uses Salesforce's BLIP model to understand what's happening
- Speaks the captions out loud
- Works best with a GPU (way faster)

## Tech used

- BLIP (Salesforce/blip-image-captioning-large) for image understanding
- PyTorch for running the AI
- CUDA for GPU support
- pyttsx3 for text-to-speech
- OpenCV and MSS for screen capture

## Requirements

- Python 3.7+
- NVIDIA GPU (recommended)
- CUDA toolkit if using GPU

## Quick start

```bash
pip install -r requirements.txt
python main.py
```


- First run downloads the BLIP model (~1GB)
- Sometimes sees everything as "screenshots" - working on fixing this
- Voice might stutter if too many captions come in fast
