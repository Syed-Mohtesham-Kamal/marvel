from doctest import master
from operator import contains
import os
import wave
import time
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
from Fuctions.online import play_on_youtube,search_on_google,send_whatsapp_message,send_email,search_on_wikipedia,get_random_joke,get_random_advice
import pyttsx3
import speech_recognition as sr
import datetime
import streamlit as st
from random import choice
from utils import opening_text
import os
from Fuctions.os_ops import open_calculator, open_camera, open_cmd
import json
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import pickle

components.html("""
   <link href="https://fonts.googleapis.com/css?family=Roboto" rel="stylesheet">
   <style>
   body {
    background-color: #fff;
    font-family: "Roboto";
  }
  
  .title {
    font-size: 32px;  
    color: #9E9E9E;
    margin: 20px 0px;
    text-align: center;
  }
  
  .canvas {
    position: relative;
    display: block;
    margin: auto;
    width: 600px;
    height: 420px;
    border-radius: 5px;
    background: none;
  }
  
  </style>
   <div class="title">M.A.R.V.E.L</div>
   
   <div class="canvas">
   </div>
""")

with open("Voice.json") as source:
    speak = json.load(source)
    
st_lottie(speak)

if st.button("Speak"):
    warnings.filterwarnings("ignore")
    st.write("Initializing Marvel")
    
    engine = pyttsx3.init('sapi5')
    engine.setProperty('rate', 190)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    def speak(text):
        engine.say(text)
        engine.runAndWait()
    
    def takeCommand():
        
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write('Listening....')
            r.pause_threshold = 1
            audio = r.listen(source)
            
            try:
                st.write('Recognizing...')
                query = r.recognize_google(audio, language='en-in')
                if not 'exit' in query or 'stop' in query:
                    speak(choice(opening_text))
                else:
                    hour = datetime.now().hour
                    if hour >= 21 and hour < 6:
                        st.write("Good night sir, take care!")
                        speak("Good night sir, take care!")
                    else:
                        st.write("Have a good day sir!")
                        speak('Have a good day sir!')
                        exit()
            except Exception:
                st.write("Sorry, I could not understand. Could you please say that again?")
                speak('Sorry, I could not understand. Could you please say that again?')                      
                query = 'None'
                st.write(query)
            return query
    
    def calculate_delta(array):

        rows, cols = array.shape
        print(rows)
        print(cols)
        deltas = np.zeros((rows, 20))
        N = 2
        for i in range(rows):
            index = []
            j = 1
            while j <= N:
                if i-j < 0:
                    first = 0
                else:
                    first = i-j
                if i+j > rows-1:
                    second = rows-1
                else:
                    second = i+j
                index.append((second, first))
                j += 1
            deltas[i] = (array[index[0][0]]-array[index[0][1]] +
                        (2 * (array[index[1][0]]-array[index[1][1]]))) / 10
        return deltas


    def extract_features(audio, rate):

        mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01,
                                20, nfft=1200, appendEnergy=True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        print(mfcc_feature)
        delta = calculate_delta(mfcc_feature)
        combined = np.hstack((mfcc_feature, delta))
        return combined


    def record_audio_train():
        speak("Hey, Please enter your name")
        st.write("Hey, Please enter your name")
        Name = st.text_input("Please enter you name:")
        speak("As a part of our verification process, we would like to take your five voice samples")
        st.write("As a part of our verification process, we would like to take your five voice samples")
        
        for count in range(5):
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            CHUNK = 512
            RECORD_SECONDS = 5
            device_index = 2
            audio = pyaudio.PyAudio()
            index = 1
            speak("recording started")
            st.write("recording started")
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, input=True, input_device_index=index,
                                frames_per_buffer=CHUNK)
            Recordframes = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                Recordframes.append(data)
            speak("recording stopped")
            st.write("recording stopped")
            stream.stop_stream()
            stream.close()
            audio.terminate()
            OUTPUT_FILENAME = Name+"-sample"+str(count)+".wav"
            WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)
            trainedfilelist = open("training_set_addition.txt", 'a')
            trainedfilelist.write(OUTPUT_FILENAME+"\n")
            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(Recordframes))
            waveFile.close()


    def record_audio_test():

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        info = audio.get_host_api_info_by_index(0)
        index = 1
        speak("Let me verify you")
        st.write("Let me verify you")
        speak("Please speak few words")
        st.write("Please speak few words...")
        speak("Recording started")
        st.write("Recording started")
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=index,
                            frames_per_buffer=CHUNK)
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        speak("Recording stopped")
        st.write("Recording stopped")
        speak("verifying...")
        st.write("verifying...")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = "sample.wav"
        WAVE_OUTPUT_FILENAME = os.path.join("testing_set", OUTPUT_FILENAME)
        trainedfilelist = open("testing_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME+"\n")
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()


    def train_model():

        source = "C:\\Users\\dell\\Desktop\\Marvel Final\\training_set\\"
        dest = "C:\\Users\\dell\\Desktop\\Marvel Final\\trained_models\\"
        train_file = "C:\\Users\\dell\\Desktop\\Marvel Final\\training_set_addition.txt\\"
        file_paths = open(train_file, 'r')
        count = 1
        features = np.asarray(())
        for path in file_paths:
            path = path.strip()
            print(path)

            sr, audio = read(source + path)
            print(sr)
            vector = extract_features(audio, sr)

            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
            
            if count == 1:
                gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init = 3)
                gmm.fit(features)

                picklefile = path.split("-")[0]+".gmm"
                pickle.dump(gmm, open(dest + picklefile, 'wb'))
                features = np.asarray(())
                count = 0
            count = count + 1


    def test_model():

        source = "C:\\Users\\dell\\Desktop\\Marvel Final\\training_set\\"
        modelpath = "C:\\Users\\dell\\Desktop\\Marvel Final\\trained_models\\"
        test_file = "C:\\Users\\dell\\Desktop\Marvel Final\\training_set_addition.txt"
        file_paths = open(test_file, 'r')

        gmm_files = [os.path.join(modelpath, fname) for fname in
                    os.listdir(modelpath) if fname.endswith('.gmm')]
        models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
        speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                    in gmm_files]

        
        for path in file_paths:

            path = path.strip()
            sr, audio = read(source + path)
            vector = extract_features(audio, sr)

            log_likelihood = np.zeros(len(models))

            for i in range(len(models)):
                gmm = models[i]
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()

            winner = np.argmax(log_likelihood)
            time.sleep(1.0)
        global MASTER
        MASTER = speakers[winner]

    def main():
        speak('Hey, I am Marvel and I am glad to meet you!')
        st.write('Hey, I am Marvel and I am glad to meet you!')
        speak('If I may ask do I know you or not?')
        st.write('If I may ask do I know you or not?')
        query = takeCommand()
        st.write(query)
        if "yes" in query:
            record_audio_test()
            test_model()
            queries()
        elif 'no'in query:
            record_audio_train()
            train_model()
            record_audio_test()
            test_model()
            queries()
        else:
            speak("Sorry, I wasn't able to hear you")
            st.write("Sorry, I wasn't able to hear you")
            speak("Well, I will take that as a NO!")
            st.write("Well, I will take that as a NO!")
            record_audio_train()
            train_model()
            record_audio_test()
            test_model()
            queries()

    def wishMe():
        hour = int(datetime.datetime.now().hour)

        if hour>=0 and hour <12:
            speak("Heyyyy,good morning" + MASTER)
            st.write("Heyyyy,good morning",MASTER)

        elif hour>=12 and hour<18:
            speak("Heyyyy,good afternoon" + MASTER)
            st.write("Heyyyy,good afternoon",MASTER)

        else:
            speak("Heyyyy,good Evening" + MASTER)
            st.write("Heyyyy,good Evening",MASTER)
        speak("i am Marvel. How may I help you?")
        st.write("i am Marvel. How may I help you?")

    def queries():
        wishMe()
        commands()

    def commands():
        query = takeCommand()
        if 'search on Wikipedia' in query:
            speak('What do you want to search on Wikipedia, sir?')
            st.write("i am Marvel. How may I help you?")
            search_query = takeCommand().lower()
            results = search_on_wikipedia(search_query)
            speak(f"According to Wikipedia, {results}")
            speak("For your convenience, I am printing it on the screen sir.")
            st.write("i am Marvel. How may I help you?")
            st.write(results)

        elif 'open command prompt' in query or 'open cmd' in query:
            open_cmd()

        elif 'open camera' in query:
            open_camera()

        elif 'open calculator' in query:
            open_calculator()

        elif 'video on YouTube' in query:
            speak("what do you want to play on Youtube, Sir?")
            st.write("what do you want to play on Youtube, Sir?")
            video = takeCommand().lower()
            play_on_youtube(video)
            
        elif 'search on Google' in query:
            speak('What do you want to search on Google, sir?')
            st.write("What do you want to search on Google, sir?")
            query = takeCommand().lower()
            search_on_google(query)
        
        elif "send WhatsApp message" in query:
            speak('On what number should I send the message sir? Please enter in the console: ')
            number = input("Enter the number: ")
            speak("What is the message sir?")
            message = takeCommand().lower()
            send_whatsapp_message(number, message)
            speak("I've sent the message sir.")

        elif 'play music' in query.lower():
            songs_dir = "C:\\Users\\dell\\Music"
            songs = os.listdir(songs_dir)
            st.write(songs)
            os.startfile(os.path.join(songs_dir, songs[1]))

        elif 'what is the time' in query.lower():
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"{MASTER} the time is {strTime}")

        
        elif "send an email" in query:
            speak("On what email address do I send sir? Please enter in the console: ")
            st.write("On what email address do I send sir? Please enter in the console: ")
            receiver_address = st.text_input("Enter email address: ")
            speak("What should be the subject sir?")
            subject = takeCommand().capitalize()
            st.write(subject)
            speak("What is the message sir?")
            message = takeCommand().capitalize()
            st.write(message)
            if send_email(receiver_address, subject, message):
                speak("I've sent the email sir.")
                st.write("I've sent the email sir.")
            else:
                speak("Something went wrong while I was sending the mail. Please check the error logs sir.")
                st.write("Something went wrong while I was sending the mail. Please check the error logs sir.")
            
        elif 'joke' in query:
            speak(f"Hope you like this one sir")
            joke = get_random_joke()
            speak(joke)
            speak("For your convenience, I am printing it on the screen sir.")
            st.write(joke)

        elif "advice" in query:
            speak(f"Here's an advice for you, sir")
            advice = get_random_advice()
            speak(advice)
            speak("For your convenience, I am printing it on the screen sir.")
            st.write(advice)
        else:
            speak("Sorry,I wasn't able to hear you" + MASTER)
            st.write("Sorry,I wasn't able to hear you", MASTER)
            commands()
    main()