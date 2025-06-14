# duygu-analizi
import streamlit as st
from deepface import DeepFace
import cv2

st.title("Duygu Analiz Uygulaması")
st.write("Kameranı ya da fotoğrafını yükle, biz de duygunu söyleyelim!")

photo = st.file_uploader("Fotoğraf yükle", type=['jpg', 'png'])

if photo:
    img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), 1)
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    st.write("Duygun :", result[0]['dominant_emotion'])
    st.image(photo, caption='Yüklediğin fotoğraf')
    
