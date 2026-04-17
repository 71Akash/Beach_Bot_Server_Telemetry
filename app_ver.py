from flask import Flask, Response
import pyrealsense2 as rs
import numpy as np
import cv2
import threading

app = Flask(__name__)

# ==========================
# RealSense Setup
# ==========================
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)

combined_frame = None
lock = threading.Lock()
running = True


def camera_loop():
    global combined_frame, running

    while running:
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # RGB
            color_image = np.asanyarray(color_frame.get_data())

            # Depth
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_scaled = cv2.convertScaleAbs(depth_image, alpha=0.08)
            depth_colormap = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

            # Combine side-by-side
            combined = np.hstack((color_image, depth_colormap))

            with lock:
                combined_frame = combined.copy()

        except Exception as e:
            print("Error:", e)


def generate():
    global combined_frame

    while True:
        with lock:
            if combined_frame is None:
                continue
            frame = combined_frame.copy()

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() +
               b'\r\n')


@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Remote Monitoring</title>
        <style>
            body {
                text-align: center;
                background-color: #111;
                color: white;
                font-family: Arial;
            }
            h1 {
                margin-top: 20px;
            }
            img {
                margin-top: 20px;
                border: 3px solid white;
                border-radius: 10px;
                max-width: 95%;
            }
        </style>
    </head>
    <body>
        <h1>REMOTE MONITORING AND SURVEILLANCE</h1>
        <img src="/video">
    </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()

    print("Server running...")
    app.run(host='0.0.0.0', port=5000, threaded=True)

    running = False
    pipeline.stop()