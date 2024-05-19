import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Khởi tạo các biến toàn cục
cap = None  # Đối tượng video capture
running = False  # Cờ để kiểm soát vòng lặp xử lý video
count = 0  # Biến đếm số lượng xe được phát hiện

# Hàm tính toán trung tâm của hình chữ nhật
def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Hàm xử lý video
def process_video(file_path):
    global cap, running, count
    min_width_react = 80  # Chiều rộng tối thiểu của hình chữ nhật
    min_height_react = 80  # Chiều cao tối thiểu của hình chữ nhật
    count_line_position = 550  # Vị trí đường đếm
    algo = cv2.bgsegm.createBackgroundSubtractorMOG()  # Thuật toán trừ nền
    detect = []  # Danh sách các điểm trung tâm của các phương tiện đã phát hiện
    offset = 6  # Độ lệch pixel để phát hiện xe vượt qua đường đếm

    cap = cv2.VideoCapture(file_path)  # Mở tệp video
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        resize_video = cv2.resize(frame, (1080, 720))  # Thay đổi kích thước khung hình
        grey = cv2.cvtColor(resize_video, cv2.COLOR_BGR2GRAY)  # Chuyển đổi khung hình sang màu xám
        blue = cv2.GaussianBlur(grey, (3, 3), 5)  # Làm mờ khung hình
        img_sub = algo.apply(blue)  # Áp dụng thuật toán trừ nền
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # Giãn nở ảnh
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Tạo kernel hình elip
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Áp dụng phép đóng
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)  # Áp dụng phép đóng lần nữa
        counterS, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Tìm các đường viền

        # Vẽ đường đếm
        cv2.line(resize_video, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

        for (i, c) in enumerate(counterS):
            (x, y, w, h) = cv2.boundingRect(c)  # Lấy hình chữ nhật bao quanh đường viền
            validate_counter = (w >= min_width_react) and (h >= min_height_react)  # Kiểm tra kích thước hình chữ nhật
            if not validate_counter:
                continue

            # Vẽ hình chữ nhật và thêm văn bản
            cv2.rectangle(resize_video, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resize_video, "Vehicle :" + str(count), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 5)

            center = center_handle(x, y, w, h)  # Tính toán trung tâm của hình chữ nhật
            detect.append(center)  # Thêm trung tâm vào danh sách
            cv2.circle(resize_video, center, 4, (0, 0, 255), -1)  # Vẽ hình tròn tại trung tâm

            # Kiểm tra nếu phương tiện đã vượt qua đường đếm
            for (cx, cy) in detect:
                if cy < (count_line_position + offset) and cy > (count_line_position - offset):
                    count += 1
                    cv2.line(resize_video, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                    detect.remove((cx, cy))  # Loại bỏ trung tâm đã vượt qua
                    print("Vehicle Counter:" + str(count))

        # Thêm văn bản tổng số phương tiện đếm được
        cv2.putText(resize_video, "Vehicle Counter :" + str(count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # Chuyển đổi khung hình thành ảnh mà Streamlit có thể sử dụng
        rgb_image = cv2.cvtColor(resize_video, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)

        # Hiển thị khung hình trong Streamlit
        st.image(img)

        if not running:
            break

    cap.release()  # Giải phóng đối tượng video capture
    running = False  # Đặt lại cờ chạy

# Thiết lập giao diện Streamlit
st.title("Vehicle Counter")

# Tải lên tệp video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    # Nút bắt đầu xử lý
    if st.button("Start Processing"):
        if running:
            running = False  # Dừng xử lý nếu đang chạy
            st.write("Processing stopped.")
        else:
            running = True  # Bắt đầu xử lý
            process_video(uploaded_file.name)
