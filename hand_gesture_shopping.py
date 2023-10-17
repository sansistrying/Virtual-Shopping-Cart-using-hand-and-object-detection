import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

# Load YOLOv3 model for object detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize video capture with the camera index (change this if needed)
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, adjust the index as necessary

# Load MediaPipe hand landmark detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define hand gesture labels
HAND_GESTURES = ["Picking up object", "Placing object back", "Placing object in virtual cart", "Placing object in shelf"]

# Initialize variables for object detection
object_detected = False
object_label = None
object_last_seen = 0
object_in_hand = False  # Track if the object is in hand
object_picked_up = False  # Track if an object is picked up
gesture_label = None
prev_object_position = None  # Track the previous position of the picked-up object
object_trajectory = []  # Store object trajectory points
bottle_count = 0  # Track the number of bottles detected

# Create and open a text file for recording actions
action_log_file = open("action_log.txt", "w")

# Create and open a text file for the cart
cart_file = open("cart.txt", "w")

# Initialize counters for objects added to and removed from the cart
objects_added_to_cart = 0
objects_removed_from_cart = 0

# Define the position of the horizontal virtual line
virtual_line_y = 0  # Adjust the y-coordinate as needed
virtual_line_x1 = 0  # Adjust the starting x-coordinate as needed
virtual_line_x2 = 0  # Adjust the ending x-coordinate as needed

# Define the reference point for the shelf (x-coordinate)
shelf_reference_x = 300  # Adjust this value based on the reference point

# Variables to track the state of the picking-up gesture
gesture_frames = 0
gesture_threshold = 20
gesture_frame_threshold = 10

# Variables to track object's position relative to the green line
object_above_line = False
object_below_line = False

# Create and open a text file for the cart
cart_file = open("cart.txt", "w")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to be 3 times bigger
    frame = cv2.resize(frame, (frame.shape[1] * 3, frame.shape[0] * 3))

    height, width, channels = frame.shape

    # Calculate the position of the horizontal virtual line at a 1:3 horizontal ratio
    virtual_line_y = height - height // 3  # 1/3 from the bottom of the frame
    virtual_line_x1 = 0
    virtual_line_x2 = width

    # Draw the green horizontal virtual line on the frame
    cv2.line(frame, (virtual_line_x1, virtual_line_y), (virtual_line_x2, virtual_line_y), (0, 255, 0), 2)

    # Object detection using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7:
                label = str(classes[class_id])
                if label == "bottle":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Count the number of bottles detected
    bottle_count = len(indexes)

    if len(indexes) > 0:
        for i in indexes.flatten():
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Display object label and confidence
            cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Set object_detected and object_label
            object_detected = True
            object_label = label
            object_last_seen = 0

            # Check if the object was picked up before
            if label == "bottle" and not object_picked_up:
                # Object is in the hand
                object_picked_up = True
                object_in_hand = True
                # Write picked up object to the action log
                action_log_file.write(f"{time.time()}: Picked up object\n")

    # Hand gesture recognition using MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            landmarks_list = []
            for point in landmarks.landmark:
                x, y = int(point.x * width), int(point.y * height)
                landmarks_list.append((x, y))

            # Calculate the distance between specific landmarks to detect gestures
            # (Add your gesture detection logic here)

            # Check if the object is in hand and update gesture detection
            if object_picked_up and object_in_hand:
                gesture_frames += 1

                if gesture_frames >= gesture_frame_threshold:
                    gesture_label = HAND_GESTURES[0]  # "Picking up object" gesture
                    gesture_frames = 0

    # Track the movement of the picked-up object (when it is in hand)
    if object_picked_up and object_in_hand:
        object_center_x = int((x + x + w) / 2)
        object_center_y = int((y + y + h) / 2)

        # Record object trajectory
        object_trajectory.append((object_center_x, object_center_y))

        # Display the object's center position on the frame
        if prev_object_position:
            cv2.arrowedLine(frame, prev_object_position, (object_center_x, object_center_y), (0, 255, 0), 2)

        # Check if the object has crossed the virtual line
        if object_center_y > virtual_line_y:
            if not object_below_line:
                object_below_line = True
                object_above_line = False
                # Object added to cart
                cv2.putText(frame, "Object Added to Cart", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Increment the counter and update the cart file
                objects_added_to_cart += 1
                cart_file.write(f"Item Added to Cart: {objects_added_to_cart}\n")
        else:
            if not object_above_line:
                object_above_line = True
                object_below_line = False
                # Object removed from cart
                cv2.putText(frame, "Object Removed from Cart", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Decrement the counter and update the cart file
                objects_removed_from_cart += 1
                cart_file.write(f"Item Removed from Cart: {objects_removed_from_cart}\n")

    # Record gesture prediction confidence for the live graph
    # Replace with your gesture prediction confidence if needed
    confidence = 0.7

    # Display the frame with the virtual line and object information
    # Display bottle count and objects added/removed from the cart
    cv2.putText(frame, f"Bottles: {bottle_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"Cart: Bottles Added: {objects_added_to_cart}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Cart: Bottles Removed: {objects_removed_from_cart}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    # Update the last seen time of the object
    if object_detected:
        object_last_seen = 0
    elif object_last_seen < 30:  # You can adjust the threshold as needed
        object_last_seen += 1
    else:
        object_detected = False
        object_label = None
        object_in_hand = False
        object_picked_up = False
        prev_object_position = None

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the action log file
action_log_file.close()

# Close the cart file
cart_file.close()

# Display the object's trajectory
object_trajectory = np.array(object_trajectory)
plt.plot(object_trajectory[:, 0], object_trajectory[:, 1])
plt.title("Object Trajectory")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.show()
