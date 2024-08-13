# Action Recognition App

This project is an Action Recognition application that uses both TensorFlow and PyTorch models for processing video files. The application allows you to upload a video, process it to detect objects and annotate actions, and then view the processed video with bounding boxes and annotations.

## Features

- **PyTorch Model:** Uses a PyTorch model for object detection and bounding box annotation.
- **TensorFlow Model:** Uses a TensorFlow model for action recognition and annotation in video frames.
- **Video Processing:** Processes uploaded videos to detect objects and annotate actions at specified intervals.
- **User Interface:** Simple Tkinter-based GUI for easy video upload, processing, and playback.

## Requirements

To run this project, you will need to have Python installed along with the following Python libraries:

- TensorFlow
- PyTorch
- OpenCV
- Roboflow
- MoviePy
- NumPy
- Pillow

You can install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt



### Code Documentation

The code is organized into functions that handle different aspects of the application, such as video processing, model predictions, and GUI interaction.

#### Key Functions

- **`predict_with_pytorch(video_file_path, model)`**
  - Description: Uses the PyTorch model to predict object bounding boxes in the video.
  - Parameters:
    - `video_file_path`: The path to the video file to be processed.
    - `model`: The PyTorch model object for prediction.
  - Returns: A dictionary containing the prediction results.

- **`predict_with_tensorflow(image, model)`**
  - Description: Uses the TensorFlow model to predict actions within video frames.
  - Parameters:
    - `image`: The current video frame to be processed.
    - `model`: The TensorFlow model object for prediction.
  - Returns: The predicted class and confidence level.

- **`annotate_and_draw_bounding_boxes(video_file_path, output_file_path, annotation_interval, display_duration, pytorch_model, tf_model)`**
  - Description: Annotates the video with bounding boxes from the PyTorch model and action classes from the TensorFlow model.
  - Parameters:
    - `video_file_path`: The path to the input video file.
    - `output_file_path`: The path where the annotated video will be saved.
    - `annotation_interval`: The interval in seconds at which annotations will be added.
    - `display_duration`: The duration in seconds for which each annotation will be displayed.
    - `pytorch_model`: The PyTorch model object for object detection.
    - `tf_model`: The TensorFlow model object for action recognition.
  - Returns: None.

- **`process_video(video_path)`**
  - Description: Orchestrates the video processing by calling `annotate_and_draw_bounding_boxes`.
  - Parameters:
    - `video_path`: The path to the video file to be processed.
  - Returns: The path to the processed video.

- **`upload_video()`**
  - Description: Opens a file dialog to allow the user to upload a video file.
  - Returns: None.

- **`play_video(video_path)`**
  - Description: Plays the processed video using OpenCV.
  - Parameters:
    - `video_path`: The path to the video file to be played.
  - Returns: None.

- **`process_uploaded_video()`**
  - Description: Handles the video processing workflow after the user uploads a video.
  - Returns: None.
"# capstone" 
