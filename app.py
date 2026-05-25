from flask import Flask, Response, request, jsonify
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time

app = Flask(__name__)

# ==========================
# YOLO CONFIG
# ==========================

WEIGHT_PATH = "combine.pt"
model = YOLO(WEIGHT_PATH)

TARGET_LABELS = ['bottle', 'metal-can', 'paper-cup']
PERSON_LABELS = ['person']

print("Loading YOLO model...")
model = YOLO(WEIGHT_PATH)
print("Model loaded successfully")

# ==========================
# RealSense Camera Setup
# ==========================
pipeline = rs.pipeline()
config = rs.config()

# RGB Stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Depth Stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)

rgb_frame_data = None
detection_frame_data = None
depth_frame_data = None
raw_depth_data = None
lock = threading.Lock()

running = True

def get_median_depth(depth_frame, x1, y1, x2, y2):
    depth_values = []

    step = 4

    for x in range(x1, x2, step):
        for y in range(y1, y2, step):
            d = depth_frame.get_distance(x, y)

            if d > 0:
                depth_values.append(d)

    if len(depth_values) == 0:
        return 0

    return np.median(depth_values)

def camera_loop():
    global rgb_frame_data, depth_frame_data, raw_depth_data, running

    while running:
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # RGB image
            frame = np.asanyarray(color_frame.get_data())

            # Run YOLO
            results = model(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False
            )[0]

            vis = frame.copy()

            # Process detections
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                names = results.names

                for box, cls, conf in zip(boxes, clss, confs):
                    x1, y1, x2, y2 = map(int, box)
                    label = names[int(cls)].lower()

                    # Only allowed classes
                    if label not in TARGET_LABELS and label not in PERSON_LABELS:
                        continue

                    # Clamp
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1] - 1, x2)
                    y2 = min(frame.shape[0] - 1, y2)

                    # Median depth
                    distance = get_median_depth(depth_frame, x1, y1, x2, y2)

                    # Color
                    if label in PERSON_LABELS:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)

                    # Draw box
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                    # Center point
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

                    # Text
                    text = f"{label} {conf:.2f} {distance:.2f}m"

                    cv2.putText(
                        vis,
                        text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )

            # Depth frame
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_scaled = cv2.convertScaleAbs(depth_image, alpha=0.08)

            depth_colormap = cv2.applyColorMap(
                depth_scaled,
                cv2.COLORMAP_JET
            )

            with lock:
                rgb_frame_data = frame.copy()
                detection_frame_data = vis.copy()
                depth_frame_data = depth_colormap.copy()
                raw_depth_data = depth_image.copy()

        except Exception as e:
            print("Camera loop error:", e)
            time.sleep(0.1)


def generate_mjpeg(feed_type="rgb"):
    global rgb_frame_data, depth_frame_data

    while True:
        frame = None

        with lock:
            if feed_type == "rgb" and rgb_frame_data is not None:
                frame = rgb_frame_data.copy()
            elif feed_type =="detection" and detection_frame_data is not None:
                frame = detection_frame_data.copy()
            elif feed_type == "depth" and depth_frame_data is not None:
                frame = depth_frame_data.copy()

        if frame is None:
            time.sleep(0.03)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() +
               b'\r\n')


@app.route("/rgb")
def rgb_feed():
    return Response(generate_mjpeg("rgb"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/depth")
def depth_feed():
    return Response(generate_mjpeg("depth"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detection")
def detection_feed():
    return Response(
        generate_mjpeg("detection"),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/depth_value")
def depth_value():
    global raw_depth_data

    try:
        x = int(request.args.get("x", 0))
        y = int(request.args.get("y", 0))

        with lock:
            if raw_depth_data is None:
                return jsonify({"error": "No depth frame available"}), 503

            h, w = raw_depth_data.shape

            if x < 0 or x >= w or y < 0 or y >= h:
                return jsonify({"error": "Coordinates out of bounds"}), 400

            depth_mm = int(raw_depth_data[y, x])
            distance_m = round(depth_mm / 1000.0, 3)

        return jsonify({
            "x": x,
            "y": y,
            "distance_m": distance_m
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return """
    <h2>RealSense Camera Server Running</h2>
    <ul>
        <li><a href="/rgb">RGB Feed</a></li>
        <li><a href="/depth">Depth Feed</a></li>
    </ul>
    """


if __name__ == "__main__":
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()

    try:
        print("Starting RealSense camera server on http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        running = False
        pipeline.stop()