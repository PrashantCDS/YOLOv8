from flask import Flask, render_template, Response, request
import cv2
import os
import base64
import math
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Initialize YOLOv8 model
model = YOLO('best.pt')

# Define folder paths
input_images_folder = "input-images"
output_images_folder = "output-images"

# Ensure the existence of the folders
os.makedirs(input_images_folder, exist_ok=True)
os.makedirs(output_images_folder, exist_ok=True)

# Function to perform object detection
def detect_objects(frame):
    results = model.predict(source=frame, conf=0.5)  # You can adjust the confidence threshold as needed
    classNames = results[0].names

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # class name
            cls = int(box.cls[0])

            # Display name, confidence, and bounding box info
            text = f"{classNames[cls]} : {confidence:.2f}"
            org = (x1, y1 - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(frame, text, org, font, fontScale, color, thickness)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Failed to open camera.")
            return

        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture frame.")
                break

            frame = detect_objects(frame)

            _, jpeg = cv2.imencode('.jpg', frame)
            data = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('index.html')

    image = request.files['image']
    if image.filename == '':
        return render_template('index.html')

    input_image_path = os.path.join(input_images_folder, image.filename)
    image.save(input_image_path)

    img = cv2.imread(input_image_path)
    output_img = detect_objects(img)

    output_image_path = os.path.join(output_images_folder, "output_" + image.filename)
    cv2.imwrite(output_image_path, output_img)

    with open(input_image_path, "rb") as file:
        input_image_base64 = base64.b64encode(file.read()).decode('utf-8')

    with open(output_image_path, "rb") as file:
        output_image_base64 = base64.b64encode(file.read()).decode('utf-8')

    return render_template('index.html', input_image=input_image_base64, output_image=output_image_base64)

if __name__ == '__main__':
    app.run(debug=True)
