import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from threading import Thread
from PIL import Image, ImageTk

# Khởi tạo các biến toàn cục
cap = None  # Đối tượng video capture
running = False  # Cờ để kiểm soát vòng lặp xử lý video
count = 0  # Biến đếm số lượng xe được phát hiện

# Hàm tính toán trung tâm của hình chữ nhật
def center_handle(x, y, w, h):
    # Tính toán trung tâm của một hình chữ nhật dựa vào góc trên bên trái (x, y) và chiều rộng (w), chiều cao (h)
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Hàm xử lý các khung hình video và phát hiện xe
def process_video():
    global cap, running, count, frame_label
    min_width_react = 80  # Chiều rộng tối thiểu của đối tượng được phát hiện để coi là xe
    min_height_react = 80  # Chiều cao tối thiểu của đối tượng được phát hiện để coi là xe
    count_line_position = 550  # Tọa độ Y của đường đếm xe
    algo = cv2.bgsegm.createBackgroundSubtractorMOG()  # Thuật toán trừ nền
    detect = []  # Danh sách lưu trữ các trung tâm của các xe được phát hiện
    offset = 6  # Độ lệch pixel để phát hiện xe vượt qua đường đếm

    while running:
        ret, frame = cap.read()  # Đọc một khung hình từ video
        if not ret:  # Nếu không có khung hình nào được trả về, kết thúc vòng lặp
            break
        
        # Thay đổi kích thước khung hình để xử lý đồng đều
        resize_video = cv2.resize(frame, (1080, 720))
        grey = cv2.cvtColor(resize_video, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
        blue = cv2.GaussianBlur(grey, (3, 3), 5)  # Áp dụng làm mờ Gaussian để giảm nhiễu
        img_sub = algo.apply(blue)  # Áp dụng thuật toán trừ nền
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # Làm giãn ảnh để lấp đầy các khoảng trống
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Phần tử cấu trúc cho các phép biến hình học
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Áp dụng phép đóng hình học
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)  # Áp dụng phép đóng hình học lần nữa
        counterS, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Tìm các đường viền

        # Vẽ đường đếm xe trên khung hình
        cv2.line(resize_video, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

        for (i, c) in enumerate(counterS):
            (x, y, w, h) = cv2.boundingRect(c)  # Lấy khung bao của đường viền
            validate_counter = (w >= min_width_react) and (h >= min_height_react)  # Kiểm tra kích thước của đường viền
            if not validate_counter:
                continue

            # Vẽ hình chữ nhật xung quanh xe được phát hiện
            cv2.rectangle(resize_video, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resize_video, "Vehicle :" + str(count), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 5)

            center = center_handle(x, y, w, h)  # Tính toán trung tâm của xe được phát hiện
            detect.append(center)  # Thêm trung tâm vào danh sách phát hiện
            cv2.circle(resize_video, center, 4, (0, 0, 255), -1)  # Vẽ một vòng tròn tại trung tâm

            for (cx, cy) in detect:
                if cy < (count_line_position + offset) and cy > (count_line_position - offset):
                    count += 1  # Tăng biến đếm xe
                    # Vẽ đường chỉ thị rằng xe đã được đếm
                    cv2.line(resize_video, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                    detect.remove((cx, cy))  # Xóa trung tâm khỏi danh sách
                    print("Vehicle Counter:" + str(count))

        # Hiển thị số lượng xe trên khung hình
        cv2.putText(resize_video, "Vehicle Counter :" + str(count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # Chuyển đổi khung hình thành ảnh mà Tkinter có thể sử dụng
        rgb_image = cv2.cvtColor(resize_video, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Cập nhật nhãn với ảnh mới
        frame_label.imgtk = imgtk
        frame_label.configure(image=imgtk)
        frame_label.image = imgtk

        if cv2.waitKey(10) == 27:  # Kiểm tra xem có nhấn phím 'Esc' để thoát không
            break

    cap.release()  # Giải phóng đối tượng video capture
    cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV
    running = False  # Đặt lại cờ chạy

# Hàm bắt đầu xử lý video
def start_processing():
    global cap, running
    if not running:  # Kiểm tra nếu xử lý chưa chạy
        file_path = 'video.mp4'  # Đường dẫn đến tệp video
        if file_path:
            cap = cv2.VideoCapture(file_path)  # Mở tệp video
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return
            running = True  # Đặt cờ chạy thành True
            process_thread = Thread(target=process_video)  # Tạo một luồng mới để xử lý video
            process_thread.start()  # Bắt đầu luồng

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Vehicle Counter")
root.geometry("1100x800")  # Đặt kích thước của cửa sổ

# Tạo nút bắt đầu
start_button = tk.Button(root, text="Start", command=start_processing)
start_button.pack(pady=20)

# Tạo nhãn để hiển thị các khung hình video
frame_label = tk.Label(root)
frame_label.pack()

# Chạy vòng lặp sự kiện của Tkinter
root.mainloop()
