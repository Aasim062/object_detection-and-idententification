# object_detection-and-idententification

In the realm of computer vision, object detection is a critical task that involves identifying instances of objects from predefined classes within an image. This paper discusses the implementation of a custom object detection system using the YOLOv3 (You Only Look Once) model. The system is tailored to detect 3D objects by employing a custom-trained YOLOv3 model and library. The primary objective is to identify and save detected objects in images captured from a webcam feed, specifically focusing on those with a white background.

Methodology
Custom YOLOv3 Model
YOLOv3 is a state-of-the-art object detection model known for its speed and accuracy. For this project, we have trained a YOLOv3 model on a custom dataset containing 3D objects. The model configuration, weights, and class names are provided in the following files:

Class Names File: Contains the names of the 3D objects to be detected.
Configuration File (cfg): Defines the structure of the YOLOv3 model.
Weights File: Stores the trained parameters of the model.
Implementation Steps
Loading Class Names:

The class names are loaded from a file (custom.names) which lists all the 3D objects the model can detect.
Model Setup:

The YOLOv3 model is initialized using the configuration and weights files (yolov3_custom.cfg and yolov3-custom_last.weights respectively).
The model is configured to accept input images of size 320x320 pixels, with normalization and mean subtraction applied to match the training conditions.
Image Capture and Preprocessing:

The system captures frames from a webcam using OpenCV.
Each frame is checked for a white background. If the majority of the border pixels are white, the frame is processed further.
Object Detection:

The model detects objects in the frame, returning bounding boxes, class IDs, and confidence scores.
Only objects with confidence scores above a specified threshold (0.6) are considered valid detections.
Annotation and Saving Detected Objects:

Detected objects are annotated on the frame with bounding boxes, class names, and confidence scores.
Objects with the highest confidence are saved as individual images, along with their detection metadata, to a designated directory.
Logging:

Detected objects and their confidence scores are logged to a text file (detected_objects.txt).
