import tensorflow
from keras.models import load_model
import librosa
import pickle
import numpy as np
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import time
import IPython.display as ipd

freq = 22050

duration = 6
# global names
# names = None


model = load_model("Neuron")
label = pickle.load(open("EmotionLabels.pkl", 'rb'))


def audioExtract(file_name):
    final = []
    audio, sam = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(
        y=audio, sr=sam, n_mfcc=40)
    mfcc_scaled = np.mean(mfccs_features.T, axis=0)
    mfcc_scaled = mfcc_scaled.reshape(1, -1)
    predicted = model.predict(mfcc_scaled)
    for i in predicted:
        final.append(np.round(i))
    d = []
    for i in label.classes_:
        d.append(i)

    output = []
    for i in predicted:
        kp = list(i)
        for i in kp:
            output.append(i)

    j = 0
    for i in range(len(output)):
        if round(output[i]) == 1:
            # print(d[i])
            st.subheader(
                f"YOUR EMOTION SOUNDS LIKE : {d[i].upper()}")
            break


def record(filename):
    with st.spinner('Started Recording... It will automatically quit after 5 secs'):
        recording = sd.rec(int(duration * freq),
                           samplerate=freq, channels=2)

        sd.wait()

        write(f"{filename}.wav", freq, recording)
        print("saved")
        st.write(f"SELECTED FILE {filename}.wav")


def stream():
    st.title('EMOTIONS RECOGNIZER')
    intro = 'DEVELOPED WITH ❤️ BY ATIF | RHEA | AND ANUSHRI'
    st.header(intro)
    st.write("*"*100)
    st.subheader("RECORD YOUR OWN AUDIO")
    user_audio_name = st.text_input('WRITE FILE NAME WITH NO EXTENSIONS: ', '')
    recording_but = st.button("START RECORDING")
    if recording_but:
        if len(user_audio_name) != 0:
            record(user_audio_name)
        else:
            st.error("FILE NAME CAN'T BE EMPTY")

    st.write("-"*100)
    st.subheader("SELECT a .wav FILE")
    uploaded_file = st.file_uploader(
        "CHOOSE A FILE", accept_multiple_files=True)
    print("uploaded :  ", len(uploaded_file))
    
    directory_name = "Audio"
    current_dir = os.getcwd()
    path = os.path.join(current_dir, directory_name)
    st.write(path)
    
    for i in uploaded_file:
        names = i.name
        st.write(f"YOU SELECTED : {names}")
        st.write(ipd.Audio(names))

    col1, col2, col3 = st.columns(3)
    user_audio_name = f"{user_audio_name}.wav"
    with col2:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        predict = st.button("EMOTION CHECK")
        if predict:
            if len(uploaded_file) != 0:
                print("File dragged")
                audioExtract(names)
            else:
                print("file recorded")
                audioExtract(user_audio_name)


stream()
