import cv2
import numpy as np
import os
import datetime

def is_white_background(img, threshold=200, border_width=50):
    h, w, _ = img.shape

    top_border = img[0:border_width, :]
    bottom_border = img[h-border_width:h, :]
    left_border = img[:, 0:border_width]
    right_border = img[:, w-border_width:w]

    gray_top = cv2.cvtColor(top_border, cv2.COLOR_BGR2GRAY)
    gray_bottom = cv2.cvtColor(bottom_border, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.cvtColor(left_border, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_border, cv2.COLOR_BGR2GRAY)

    _, binary_top = cv2.threshold(gray_top, threshold, 255, cv2.THRESH_BINARY)
    _, binary_bottom = cv2.threshold(gray_bottom, threshold, 255, cv2.THRESH_BINARY)
    _, binary_left = cv2.threshold(gray_left, threshold, 255, cv2.THRESH_BINARY)
    _, binary_right = cv2.threshold(gray_right, threshold, 255, cv2.THRESH_BINARY)

    white_ratio_top = np.sum(binary_top == 255) / binary_top.size
    white_ratio_bottom = np.sum(binary_bottom == 255) / binary_bottom.size
    white_ratio_left = np.sum(binary_left == 255) / binary_left.size
    white_ratio_right = np.sum(binary_right == 255) / binary_right.size

    return (white_ratio_top > 0.8 or white_ratio_bottom > 0.8 or
            white_ratio_left > 0.8 or white_ratio_right > 0.8)

def detect_objects(img, thres, nms, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    detected_objects = []
    if len(objects) == 0:
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects and confidence >= 0.6:
                detected_objects.append((box, className, confidence))
    return detected_objects

def draw_and_save_objects(img, detected_objects, save_path, saved_objects):
    max_confidence = 0
    max_confidence_box = None
    max_confidence_class = None

    for box, className, confidence in detected_objects:
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        if confidence > max_confidence:
            max_confidence = confidence
            max_confidence_box = box
            max_confidence_class = className

    if max_confidence_box is not None and max_confidence_class not in saved_objects:
        x, y, w, h = max_confidence_box
        object_img = img[y:y+h, x:x+w]
        current_datetime = datetime.datetime.now()
        folder_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        image_name = f"{max_confidence_class}_{current_datetime.strftime('%Y%m%d_%H%M%S')}_confidence_{int(max_confidence * 100)}.jpg"
        cv2.imwrite(os.path.join(folder_path, image_name), object_img)
        saved_objects.add(max_confidence_class)
    return img

def load_class_names(file_path):
    try:
        with open(file_path, "rt") as f:
            return f.read().rstrip("\n").split("\n")
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        exit(1)

def setup_detection_model(config_path, weights_path):
    try:
        net = cv2.dnn_DetectionModel(weights_path, config_path)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        return net
    except cv2.error as e:
        print(f"Error loading model: {e}")
        exit(1)

if __name__ == "__main__":
    classFile = "Object_Detection_Files/coco.names"
    configPath = "Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "Object_Detection_Files/frozen_inference_graph.pb"

    classNames = load_class_names(classFile)
    net = setup_detection_model(configPath, weightsPath)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    total_objects_detected = 0
    unique_objects = set()
    save_path = "detected_objects_images"
    os.makedirs(save_path, exist_ok=True)
    saved_objects = set()

    with open('detected_objects.txt', 'w') as file:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            detected_objects = detect_objects(frame, 0.45, 0.2)

            if detected_objects:
                if is_white_background(frame):
                    frame = draw_and_save_objects(frame, detected_objects, save_path, saved_objects)
                    for _, class_name, confidence in detected_objects:
                        if class_name not in unique_objects:
                            if confidence >= 0.6:
                                total_objects_detected += 1
                                unique_objects.add(class_name)
                                file.write(f"{class_name} - Confidence: {confidence}\n")

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print("Total unique objects detected:", total_objects_detected)
    print("Different objects detected:", unique_objects)

    cap.release()
    cv2.destroyAllWindows()
