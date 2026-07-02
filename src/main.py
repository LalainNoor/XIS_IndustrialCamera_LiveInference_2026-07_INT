import cv2
from camera import Camera
import camera
from config import *
import metrics
from preprocess import preprocess
from inference import ONNXClassifier

import time

from display import draw_overlay
from metrics import PerformanceMetrics

def main():

    camera = Camera(VIDEO_PATH)

    print("=" * 60)
    print("Industrial Camera Live Inference")
    print("=" * 60)

    print(f"Video Source : {VIDEO_PATH}")
    print(f"Resolution   : {camera.get_width()} x {camera.get_height()}")
    print(f"FPS          : {camera.get_fps():.2f}")
    print(f"Frame Count  : {camera.get_frame_count()}")
    print("=" * 60)

    classifier = ONNXClassifier(
    MODEL_PATH,
    CLASS_NAMES
    )

    metrics = PerformanceMetrics()

    while True:

        ret, frame = camera.read()

        if not ret:
            print("End of video.")
            break

        start = time.perf_counter()

        input_tensor = preprocess(frame)
        prediction, confidence = classifier.predict(input_tensor)

        latency = metrics.calculate_latency(start)
        metrics.update(latency)
        fps = metrics.calculate_fps()
        throughput = fps

        frame = draw_overlay(
            frame,
            prediction,
            confidence,
            fps,
            latency,
            throughput,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        summary = metrics.summary()

        with open("results/performance/performance.txt", "w") as f:

            f.write("Industrial Camera Live Inference\n")
            f.write("=" * 40 + "\n")

            f.write(f"Video Source : {VIDEO_PATH}\n")
            f.write(f"Resolution   : {camera.get_width()} x {camera.get_height()}\n")
            f.write(f"Total Frames     : {summary['frames']}\n")
            f.write(f"Processing Time  : {summary['elapsed']:.2f} sec\n")
            f.write(f"Average FPS      : {summary['avg_fps']:.2f}\n")
            f.write(f"Average Latency  : {summary['avg_latency']:.2f} ms\n")
            f.write(f"Throughput       : {summary['throughput']:.2f} FPS\n")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()