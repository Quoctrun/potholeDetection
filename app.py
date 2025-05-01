import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
from ultralytics import YOLO  # Sử dụng ultralytics thay vì torch.hub
from video_predict import runVideo


# Configurations
CFG_MODEL_PATH = os.path.join("models", "best.pt")  # Đường dẫn đến mô hình YOLOv11n
CFG_ENABLE_VIDEO_PREDICTION = True
# End of Configurations


def imageInput(model, src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader(
            "Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image',
                         use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data', 'uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data', 'outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            with st.spinner(text="Predicting..."):
                # Dự đoán với YOLOv11
                results = model.predict(imgpath, device=deviceoption, verbose=False)
                # Lưu kết quả dự đoán
                results[0].save(outputpath)

            # Hiển thị kết quả
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)',
                         use_column_width='always')

    elif src == 'From example data.':
        imgpaths = glob.glob('data/example_images/*')
        if len(imgpaths) == 0:
            st.write(".")
            st.error(
                'No images found, Please upload example images in data/example_images', icon="")
            return
        imgsel = st.slider('Select random images from example data.',
                           min_value=1, max_value=len(imgpaths), step=1)
        image_file = imgpaths[imgsel-1]
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                with st.spinner(text="Predicting..."):
                    # Dự đoán với YOLOv11
                    results = model.predict(image_file, device=deviceoption, verbose=False)
                    # Lưu kết quả dự đoán
                    output_path = os.path.join('data', 'outputs', os.path.basename(image_file))
                    results[0].save(output_path)
                # Hiển thị kết quả
                img_ = Image.open(output_path)
                st.image(img_, caption='Model Prediction(s)')


def videoInput(model, src, device):
    if src == 'Upload your own data.':
        uploaded_video = st.file_uploader(
            "Upload A Video", type=['mp4', 'mpeg', 'mov'])
        pred_view = st.empty()
        warning = st.empty()
        if uploaded_video is not None:
            # Lưu video
            ts = datetime.timestamp(datetime.now())
            uploaded_video_path = os.path.join(
                'data', 'uploads', str(ts) + uploaded_video.name)
            with open(uploaded_video_path, mode='wb') as f:
                f.write(uploaded_video.read())

            # Hiển thị video đã tải lên
            with open(uploaded_video_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.write("Uploaded Video")
            submit = st.button("Run Prediction")
            if submit:
                runVideo(model, uploaded_video_path, pred_view, warning, device)  # Truyền device

    elif src == 'From example data.':
        videopaths = glob.glob('data/example_videos/*')
        if len(videopaths) == 0:
            st.error(
                'No videos found, Please upload example videos in data/example_videos', icon="⚠️")
            return
        imgsel = st.slider('Select random video from example data.',
                           min_value=1, max_value=len(videopaths), step=1)
        pred_view = st.empty()
        video = videopaths[imgsel-1]
        submit = st.button("Predict!")
        if submit:
            runVideo(model, video, pred_view, warning, device)  # Truyền device


def main():
    # Kiểm tra xem file mô hình có tồn tại không
    if not os.path.exists(CFG_MODEL_PATH):
        st.error(
            f'Model file "{CFG_MODEL_PATH}" not found. Please ensure the YOLOv11n model file (e.g., "best.pt") is uploaded to the "models/" directory.', icon="⚠️")
        st.stop()

    # -- Sidebar
    st.sidebar.title('⚙️ Options')
    datasrc = st.sidebar.radio("Select input source.", [
                               'From example data.', 'Upload your own data.'])

    if CFG_ENABLE_VIDEO_PREDICTION:
        option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    else:
        option = st.sidebar.radio("Select input type.", ['Image'])

    global deviceoption
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=True, index=0)
    # -- End of Sidebar

    st.header('Pothole detection')
    st.sidebar.markdown("")

    if option == "Image":
        imageInput(loadmodel(deviceoption), datasrc)
    elif option == "Video":
        videoInput(loadmodel(deviceoption), datasrc, deviceoption)


@st.cache_resource
def loadmodel(device):
    try:
        # Tải mô hình YOLOv11n bằng ultralytics
        model = YOLO(CFG_MODEL_PATH)
        model.to(device)
        return model
    except Exception as e:
        st.error(
            f'Failed to load the YOLOv11n model: {str(e)}. Please ensure the model file is valid.', icon="⚠️")
        st.stop()


if __name__ == '__main__':
    main()