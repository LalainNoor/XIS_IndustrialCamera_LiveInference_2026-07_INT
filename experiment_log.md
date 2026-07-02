# Experiment Log

## Date
2026-07-02

## Objective
Develop a simulated industrial camera live inference pipeline using a prerecorded video source and ONNX Runtime.

## Environment
- Python Virtual Environment
- OpenCV
- ONNX Runtime
- NumPy
- Windows

## Implementation Summary

### Camera Module
- Implemented a modular camera interface using OpenCV VideoCapture.
- Simulated an industrial camera using a prerecorded video.

### Preprocessing
- Converted frames from BGR to RGB.
- Resized frames to 224 × 224.
- Normalized pixel values.
- Converted images to CHW format.

### Inference
- Loaded ConvNeXt V2 ONNX model.
- Performed frame-by-frame inference.
- Generated predictions with confidence scores.

### Performance
- Measured:
  - Average FPS
  - Average Inference Latency
  - Throughput

### Results
- Successful live inference on video stream.
- Performance report generated automatically.
- Live prediction overlay displayed on each frame.

## Notes

This project simulates an industrial GenICam camera using a prerecorded video source. The Camera module was designed to allow future replacement with a Harvester-based implementation without modifying the inference pipeline.