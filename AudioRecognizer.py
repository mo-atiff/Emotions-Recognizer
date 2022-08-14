from keras.models import load_model
import librosa
import librosa.display
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import sounddevice as sd
from scipy.io.wavfile import write
import IPython.display as ipd
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from datetime import datetime

freq = 22050

duration = 6


model = load_model("C:\\Users\\ATIF SHAIK\\Neuron")
label = pickle.load(open("C:\\Users\\ATIF SHAIK\\EmotionLabels.pkl", 'rb'))


st.set_page_config(
    page_title="Audio Emotion Recognizer",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
    #     menu_items={
    #         'Get Help': 'https://www.extremelycoolapp.com/help',
    #         'Report a bug': "https://www.extremelycoolapp.com/bug",
    #         'About': "# This is a header. This is an *extremely* cool app!"
    #     }
)


def save_audio(file):
    for i in file:
        if i.size > 4000000:
            return 1

        if not os.path.exists("Audio"):
            os.makedirs("Audio")

        folder = "Audio"

        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        # clear the folder to avoid storage overload
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        try:
            with open("log0.txt", "a") as f:
                f.write(f"{i.name} - {i.size} - {datetoday};\n")
        except:
            pass

        with open(os.path.join(folder, i.name), "wb") as f:
            f.write(i.getbuffer())
#         st.write(f"saved in : {file_path}")
        return 0


def audioExtract(file_name, model):
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

    # print(d)

    output = []
    for i in predicted:
        kp = list(i)
        for i in kp:
            output.append(i)

    j = 0
    for i in range(len(output)):
        if round(output[i]) == 1:
            st.subheader("YOUR EMOTION : ")
            # st.subheader(
            # f"EMOTION : {d[i].upper()}")
            # st.subheader(f"    {d[i].upper()}")
            myemotion = d[i].upper()
            print("idhar")

            if myemotion == "ANGRY":
                st.markdown("<h2 style='text-align: centre; color: red;'>ANGRY</h2>",
                            unsafe_allow_html=True)

            elif myemotion == "DISGUST":
                st.markdown("<h2 style='text-align: centre; color: purple;'>DISGUIST</h2>",
                            unsafe_allow_html=True)

            elif myemotion == "FEAR":
                print("911 was bad")
                st.markdown("<h2 style='text-align: centre; color: brown;'>FEAR</h2>",
                            unsafe_allow_html=True)

            elif myemotion == "HAPPY":
                st.markdown("<h2 style='text-align: centre; color: yellow;'>HAPPY</h2>",
                            unsafe_allow_html=True)

            elif myemotion == "NEUTRAL":
                st.markdown("<h2 style='text-align: centre; color: blue;'>NEUTRAL</h2>",
                            unsafe_allow_html=True)

            elif myemotion == "PLEASANT":
                st.markdown("<h2 style='text-align: centre; color: white;'>PLEASANT</h2>",
                            unsafe_allow_html=True)

            elif myemotion == "SAD":
                st.markdown("<h2 style='text-align: centre; color: grey;'>SAD</h2>",
                            unsafe_allow_html=True)

            st.write(' ')
            st.write(' ')
            st.write(' ')
            break


def record(filename):
    with st.spinner(f'Started Recording... It will automatically quit after 5 secs'):
        recording = sd.rec(int(duration * freq),
                           samplerate=freq, channels=2)

        sd.wait()
        kp = f"{filename}.wav"
        write(os.path.join("Audio", kp), freq, recording)
        print("saved")
        st.write(f"SELECTED FILE : {filename}.wav")
        # kp = f"{filename}.wav"
        st.write(ipd.Audio(os.path.join("Audio", kp)))
        st.write("SCROLL DOWN AND CLICK EMOTIONS BUTTON (⬇)")


def stream():
    global flag
    flag = 0
    st.markdown("<h1 style='text-align: centre; color: red;'>EMOTIONS RECOGNIZER 😡 😖 😱 😄 😐 😔 😮</h1>",
                unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: centre; color: white;'>DEVELOPED AS 🤖 BY ATIF RHEA AND ANUSHRI</h1>",
                unsafe_allow_html=True)
# -------------------------------------------------------------------------------------------------------------------------------------------
    st.write("-"*100)
    st.markdown("<h3 style='text-align: centre; color: cyan;'>RECORD YOUR OWN AUDIO</h3>",
                unsafe_allow_html=True)
    user_audio_name = st.text_input(
        'WRITE FILE NAME WITH NO EXTENSIONS: ', '')

    recording_but = st.button("START RECORDING ⏺️")
    if recording_but:
        if len(user_audio_name) != 0:
            record(user_audio_name)
            flag = 1
        else:
            st.error("FILE NAME CAN'T BE EMPTY")
# ---------------------------------------------------------------------------------------------------------------------------------------------
    st.write("-"*100)
    st.markdown("<h3 style='text-align: centre; color: cyan;'>SELECT a .wav FILE</h3>",
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "CHOOSE A FILE (Do not select multiple files)", accept_multiple_files=True)
    print("uploaded :  ", len(uploaded_file))
    save_audio(uploaded_file)
    for i in uploaded_file:
        names = i.name
        st.write(f"YOU SELECTED : {names}")
#         st.write(ipd.Audio(os.path.join("audio\\\{}".format(names))))
        st.write(ipd.Audio(os.path.join("Audio", names)))
        flag = 2
# -----------------------------------------------------------------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    # user_audio_name = "Audio\\\{}.wav".format(user_audio_name)
    user_audio_namey = f"{user_audio_name}.wav"
    # user_audio_names = os.path.join("Audio", user_audio_name)
    with col3:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        predict = st.button("EMOTION CHECK 💘")
        if predict:
            if len(uploaded_file) != 0:
                print("File dragged")
                # audioExtract("Audio\\\{}".format(names))
                audioExtract(os.path.join("Audio", names), model)
            else:
                print("file recorded")
                audioExtract(os.path.join("Audio", user_audio_namey), model)
                # audioExtract(os.path.join("Audio", ), user_audio_name)

    def plotextract(filename):
        y, sr = librosa.load(filename, res_type='kaiser_fast')
        return y

    def plotspec(filename):
        y, sr = librosa.load(filename, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=40)
        return mfccs_features

    col11, col22 = st.columns(2)
    print("Flag : ", flag)
    if predict:
        if flag == 0:
            print("iam here")
            with col11:
                kp = plotextract(os.path.join("Audio", user_audio_namey))
                st.subheader("YOUR VOICE")
                st.line_chart(kp)
            with col22:
                pk = plotspec(os.path.join("Audio", user_audio_namey))
                fig, ax = plt.subplots()
                cax = ax.imshow(pk, interpolation='nearest',
                                cmap=cm.coolwarm, origin='lower')
                ax.set_title('MFCC')
                ax.set_xlabel('Time')
                st.subheader("Mel-frequency cepstral coefficients".upper())
                st.write(fig)

        elif flag == 2:
            print("in2 iam here")
            with col11:
                kp = plotextract(os.path.join("Audio", names))
                st.subheader("YOUR VOICE")
                st.line_chart(kp)
            with col22:
                pk = plotspec(os.path.join("Audio", names))
                fig, ax = plt.subplots()
                cax = ax.imshow(pk, interpolation='nearest',
                                cmap=cm.coolwarm, origin='lower')
                ax.set_title('MFCC')
                ax.set_xlabel('Time')
                st.subheader("Mel-frequency cepstral coefficients".upper())
                st.write(fig)


stream()
