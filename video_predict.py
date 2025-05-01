import cv2
import streamlit as st
import os

# Đường dẫn tuyệt đối đến ffmpeg.exe
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"  # Thay đổi nếu đường dẫn của bạn khác

# Thêm đường dẫn FFmpeg vào biến môi trường PATH của Python
if os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)
else:
    st.error("FFmpeg not found at the specified path. Please ensure FFmpeg is installed correctly.", icon="⚠️")
    st.stop()

# Cấu hình OpenCV để sử dụng FFmpeg
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

def runVideo(model, video_path, pred_view, warning, device):
    # Kiểm tra FFmpeg trước khi xử lý video
    if not os.path.exists(FFMPEG_PATH):
        warning.error("FFmpeg executable not found. Video processing will fail.", icon="⚠️")
        return

    cap = cv2.VideoCapture(video_path)
    warning.warning("Prediction in progress, please wait...", icon="⚠️")

    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Tạo video writer để lưu kết quả
    output_path = video_path.replace('uploads', 'outputs').replace('.mp4', '_out.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Dự đoán với YOLOv11
        results = model.predict(frame, device=device, verbose=False)
        annotated_frame = results[0].plot()  # Sử dụng .plot() để vẽ kết quả dự đoán

        # Ghi frame đã xử lý vào video đầu ra
        writer.write(annotated_frame)

        # Hiển thị frame trong Streamlit
        pred_view.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()
    writer.release()

    # Hiển thị video kết quả
    with open(output_path, 'rb') as f:
        pred_view.video(f.read())
    warning.empty()