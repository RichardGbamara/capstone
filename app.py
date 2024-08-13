import os
import cv2
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from roboflow import Roboflow
import time
import tensorflow as tf
import torch

# Define necessary variables
SEQUENCE_LENGTH = 20  # Define SEQUENCE_LENGTH as per your requirement
IMAGE_HEIGHT = 64  # Replace with the height your model expects
IMAGE_WIDTH = 64  # Replace with the width your model expects
ANNOTATION_INTERVAL = 11  # Annotation interval in seconds
DISPLAY_DURATION = 9  # Duration to display the annotation in seconds
CLASSES_LIST = ["talking", "talking", "girraffing", "phone", "exchange_paper"]  # Replace with your actual class names

# Initialize Roboflow (Pytorch model)
rf = Roboflow(api_key="ScO43zOJgjQW88effK2P")
project = rf.workspace().project("final_work-s0dht")
pytorch_model = project.version("1").model

# Load TensorFlow model
tf_model_path = "path/to/your/tensorflow/model"  # Update with your actual TensorFlow model path
tf_model = tf.keras.models.load_model(tf_model_path)

def predict_with_pytorch(video_file_path, model):
    '''
    Predict using the PyTorch model
    '''
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            job_id, signed_url, expire_time = model.predict_video(
                video_file_path,
                fps=25,
                prediction_type="batch-video",
            )

            # Wait for results
            results = model.poll_until_video_results(job_id)
            return results
        except Exception as e:
            print(f"Error during video prediction: {e}")
            if attempt < retry_attempts - 1:
                print("Retrying...")
                time.sleep(5)
            else:
                raise

def predict_with_tensorflow(image, model):
    '''
    Predict using the TensorFlow model
    '''
    # Preprocess image
    image_resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)

    # Perform prediction
    predictions = model.predict(image_expanded)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return CLASSES_LIST[predicted_class], np.max(predictions)

def annotate_and_draw_bounding_boxes(video_file_path, output_file_path, annotation_interval, display_duration, pytorch_model, tf_model):
    '''
    This function will annotate the video with class names at specified intervals and draw bounding boxes.
    '''
    # Predict using PyTorch model
    results = predict_with_pytorch(video_file_path, pytorch_model)

    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    annotation_frame_interval = int(fps * annotation_interval)
    display_frame_duration = int(fps * display_duration)
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (original_video_width, original_video_height))
    frame_count = 0
    class_index = 0

    frame_results = results['final_work-s0dht']  # Adjust based on actual results structure

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        # Draw bounding boxes using PyTorch model
        if frame_count < len(frame_results) and 'predictions' in frame_results[frame_count]:
            frame_data = frame_results[frame_count]['predictions']
            for obj in frame_data:
                x = int(obj['x'] - obj['width'] / 2)
                y = int(obj['y'] - obj['height'] / 2)
                x2 = int(x + obj['width'])
                y2 = int(y + obj['height'])
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj['class']} {obj['confidence']:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Predict class using TensorFlow model
        predicted_class, confidence = predict_with_tensorflow(frame, tf_model)

        # Add annotations using TensorFlow model
        if frame_count % annotation_frame_interval < display_frame_duration:
            cv2.putText(frame, f"{predicted_class} {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if frame_count % annotation_frame_interval == 0:
            class_index += 1

        video_writer.write(frame)
        frame_count += 1

    video_reader.release()
    video_writer.release()

def process_video(video_path):
    output_video_path = f"annotated_{os.path.basename(video_path)}"
    annotate_and_draw_bounding_boxes(video_path, output_video_path, ANNOTATION_INTERVAL, DISPLAY_DURATION, pytorch_model, tf_model)
    return output_video_path

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if file_path:
        uploaded_video_label.config(text=f"Uploaded Video: {file_path}")
        uploaded_video_label.file_path = file_path

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", 640, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_uploaded_video():
    if hasattr(uploaded_video_label, 'file_path'):
        video_path = uploaded_video_label.file_path
        messagebox.showinfo("Processing", "Video processing started. This may take a while...")
        output_path = process_video(video_path)
        if os.path.exists(output_path):
            messagebox.showinfo("Success", "Video processed successfully!")
            play_video(output_path)
        else:
            messagebox.showerror("Error", "Error processing video. Please try again.")
    else:
        messagebox.showwarning("No Video", "Please upload a video first.")

# Create the main window
root = tk.Tk()
root.title("Action Recognition App")

upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=10)

uploaded_video_label = tk.Label(root, text="No video uploaded")
uploaded_video_label.pack(pady=5)

process_button = tk.Button(root, text="Process Video", command=process_uploaded_video)
process_button.pack(pady=10)

root.mainloop()
