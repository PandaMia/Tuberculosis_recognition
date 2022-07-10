import os
import streamlit as st
import torch
from cnn_model import CNN
from inference import ImageRecognition


device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = 'source/model/model_weights.pth'
model = CNN()
model.load_state_dict(torch.load(weights, map_location=device))
recognizer = ImageRecognition(model)

extensions = ['.jpg', '.png', '.bmp', '.gif', '.tif',
              '.jp2', '.pcx', '.ppm', '.tga']

path_dict = {'Clear 1': './source/data/examples/Normal/1.png',
             'Clear 2': './source/data/examples/Normal/2.png',
             'Clear 3': './source/data/examples/Normal/3.png',
             'Tuberculosis 1': './source/data/examples/Tuberculosis/1.png',
             'Tuberculosis 2': './source/data/examples/Tuberculosis/2.jpg',
             'Tuberculosis 3': './source/data/examples/Tuberculosis/3.jpg'}

m = st.markdown("""
<style>
div.stButton > button:first-child {
    height: 4em;
    width: 19em; 
}
</style>""", unsafe_allow_html=True)

st.title('Tuberculosis recognition')
st.text('Choose an image and run recognize')


imageLocation = st.empty()

choosed_way = st.sidebar.selectbox(
    'Upload image or choose a preset',
    ('Not choosen', 'Upload image', 'Choose a preset'))

if choosed_way == 'Upload image':
    uploaded_file = st.sidebar.file_uploader("Upload an image")
    if uploaded_file is not None:
        _, ext = os.path.splitext(uploaded_file.name)
        if ext in extensions:
            bytes_data = uploaded_file.getvalue()
            imageLocation.image(bytes_data, width=500)
            recognize = st.sidebar.button("Recognize!")
            if recognize:
                image = recognizer(bytes_data)
                imageLocation.image(image, width=500)
        else:
            st.error('Wrong extension! Please upload an image (jpg, png, bmp, gif, tif, jp2, pcx, ppm, tga)')

elif choosed_way == 'Choose a preset':
    selected_image = st.sidebar.selectbox(
        'Сhoose a preset image',
        ('Not choosen', 'Clear 1', 'Clear 2', 'Clear 3',
         'Tuberculosis 1', 'Tuberculosis 2', 'Tuberculosis 3'))
    if selected_image != 'Not choosen':
        image_path = path_dict[selected_image]
        file = open(image_path, "rb")
        bytes_data = file.read()
        imageLocation.image(bytes_data, width=500)
        recognize = st.sidebar.button("Recognize!")
        if recognize:
            image = recognizer(bytes_data)
            imageLocation.image(image, width=500)
