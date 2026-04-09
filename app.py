from flask import Flask, Response
import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time

app = Flask(__name__)

# ==========================
# RealSense Camera Setup
# ==========================
pipeline = rs.pipeline()
config = rs.config()

# RGB Stream
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Depth Stream
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)

rgb_frame_data = None
depth_frame_data = None
lock = threading.Lock()

running = True


def camera_loop():
    global rgb_frame_data, depth_frame_data, running

    while running:
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # RGB frame
            color_image = np.asanyarray(color_frame.get_data())

            # Depth frame → colorized for browser display
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.08),
                cv2.COLORMAP_JET
            )

            with lock:
                rgb_frame_data = color_image.copy()
                depth_frame_data = depth_colormap.copy()

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