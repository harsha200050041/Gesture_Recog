import mediapipe as mp
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode



result_dict = {
    "Gesture Category": 0,
    "Gesture Score": 0,
    "Handedness Category": 0,
    "Handedness Score": 0,
    "Handedness Display Name": 0,
    "Landmarks":[]
}

for i in range(21):
    landmark_obj = {"X": 0, "Y": 0, "Z": 0, "Visibility": 0, "Presence": 0}
    result_dict["Landmarks"].append(landmark_obj)

def map_landmark_to_pixel(landmark, img_width, img_height):
    x = int(landmark['X'] * img_width)
    y = int(landmark['Y'] * img_height)
    return x, y


result_dict = {
    "Gesture Category": 0,
    "Gesture Score": 0,
    "Handedness Category": 0,
    "Handedness Score": 0,
    "Handedness Display Name": 0,
    "Landmarks":[],
}

for i in range(21):
    result_dict["Landmarks"].append({"X":0,"Y":0})
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(result_dict["Landmarks"][0]["X"])
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

def print_result(result: GestureRecognizerResult, frame: mp.Image, timestamp_ms: int):
    # Initialize the dictionary with initial values
    # result_dict = initial_values.copy()

    # Extract gestures
    gestures = result.gestures
    if gestures:
        for gesture in gestures[0]:
            result_dict["Gesture Category"] = gesture.category_name
            result_dict["Gesture Score"] = gesture.score

    # Extract handedness
    handedness = result.handedness
    if handedness:
        for hand in handedness[0]:
            result_dict["Handedness Category"] = hand.category_name
            result_dict["Handedness Score"] = hand.score
            result_dict["Handedness Display Name"] = hand.display_name

    # Extract hand landmarks
    hand_landmarks = result.hand_landmarks

    if hand_landmarks:
        ha = 0
        for landmark in hand_landmarks[0]:
            print("----------------------")
            
            print("-----------------------")
            
            result_dict["Landmarks"][ha]["X"] = landmark.x
            result_dict["Landmarks"][ha]["Y"] = landmark.y
            ha+=1



# Initialize gesture recognizer options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Create a gesture recognizer instance
with GestureRecognizer.create_from_options(options) as recognizer:
    # Open the camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_timestamp_ms_int = int(frame_timestamp_ms)
        if not ret:
            break

        # Create an MP Image from the OpenCV frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Recognize gestures asynchronously
        recognizer.recognize_async(mp_image, frame_timestamp_ms_int)

        print(result_dict)
        scale_factor = 1.0

        # # Convert coordinates to integers
        # landmark_x = int(result_dict['Landmark X'] * scale_factor)
        # landmark_y = int(result_dict['Landmark Y'] * scale_factor)
        img_height, img_width, _ = frame.shape

        for landmark in result_dict['Landmarks']:
            x, y = map_landmark_to_pixel(landmark, img_width, img_height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at each landmark


        # Display the gesture category on the frame
        cv2.putText(frame, f"Gesture: {result_dict['Gesture Category']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Hand Gesture Recognition', frame)
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()
