import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from threading import Thread
from PIL import Image, ImageTk

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

def process_video():
    global cap, running, count, frame_label
    min_width_react = 80
    min_height_react = 80
    learning_rate = 1.0 / 100
    count_line_position = 550
    algo = cv2.bgsegm.createBackgroundSubtractorMOG()
    detect = []
    offset = 6  # Do sai tren pixel

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

        # Convert the frame to an image that Tkinter can use
        rgb_image = cv2.cvtColor(resize_video, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the label with the new image
        frame_label.imgtk = imgtk
        frame_label.configure(image=imgtk)
        frame_label.image = imgtk

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    running = False

def start_processing():
    global cap, running
    if not running:
        file_path = 'video.mp4'
        if file_path:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return
            running = True
            process_thread = Thread(target=process_video)
            process_thread.start()

# Create the main window
root = tk.Tk()
root.title("Vehicle Counter")
root.geometry("1100x800")

# Create buttons
start_button = tk.Button(root, text="Start", command=start_processing)
start_button.pack(pady=20)

# Create a label to display the video
frame_label = tk.Label(root)
frame_label.pack()

# Run the Tkinter event loop
root.mainloop()
