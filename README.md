# Industrial Camera Live Inference

## Overview

This project demonstrates a simulated industrial camera live inference pipeline using OpenCV and ONNX Runtime.

Since GenICam hardware was unavailable during development, a prerecorded video was used to simulate the industrial camera feed, following guidance from the project supervisor.

The project was designed so that the video source can later be replaced by a GenICam camera using the Harvester library with minimal code changes.

---

## Features

- Modular camera interface
- Video stream acquisition
- Frame preprocessing
- ConvNeXt V2 ONNX inference
- Live prediction overlay
- FPS measurement
- Inference latency measurement
- Throughput calculation
- Performance report generation

---

## Project Structure

```text
src/
camera.py
config.py
display.py
inference.py
main.py
metrics.py
preprocess.py
```

---

## Model

ConvNeXt V2

Classes

- buildings
- forest
- glacier
- mountain
- sea
- street

---

## Results

Performance metrics are automatically saved to:

```
results/performance/performance.txt
```

Screenshots are stored in:

```
results/screenshots/
```

---

## Run

```bash
python src/main.py
```

---

## Future Work

- Replace video source with GenICam industrial camera.
- Integrate Harvester SDK.
- Enable GPU inference using CUDA/TensorRT.