import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Initialize global variables
cap = None
running = False
count = 0

# Function to handle center calculation
def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Process video function
def process_video(file_path):
    global cap, running, count
    min_width_react = 80
    min_height_react = 80
    count_line_position = 550
    algo = cv2.bgsegm.createBackgroundSubtractorMOG()
    detect = []
    offset = 6  # Do sai tren pixel

    cap = cv2.VideoCapture(file_path)
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        resize_video = cv2.resize(frame, (1080, 720))
        grey = cv2.cvtColor(resize_video, cv2.COLOR_BGR2GRAY)
        blue = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = algo.apply(blue)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        counterS, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(resize_video, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

        for (i, c) in enumerate(counterS):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_counter = (w >= min_width_react) and (h >= min_height_react)
            if not validate_counter:
                continue

            cv2.rectangle(resize_video, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resize_video, "Vehicle :" + str(count), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 5)

            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(resize_video, center, 4, (0, 0, 255), -1)

            for (cx, cy) in detect:
                if cy < (count_line_position + offset) and cy > (count_line_position - offset):
                    count += 1
                    cv2.line(resize_video, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                    detect.remove((cx, cy))
                    print("Vehicle Counter:" + str(count))

        cv2.putText(resize_video, "Vehicle Counter :" + str(count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # Convert the frame to an image that Streamlit can use
        rgb_image = cv2.cvtColor(resize_video, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)

        # Display the frame in Streamlit
        st.image(img)

        if not running:
            break

    cap.release()
    running = False

# Streamlit UI setup
st.title("Vehicle Counter")

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    # Start processing button
    if st.button("Start Processing"):
        if running:
            running = False
            st.write("Processing stopped.")
        else:
            running = True
            process_video(uploaded_file.name)
