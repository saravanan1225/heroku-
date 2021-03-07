# - *- coding: utf- 8 - *-
import streamlit as st
import pandas as pd
import hashlib
import sqlite3
import numpy as np

import speech_recognition as sr
import speech_recognition as sp
import paralleldots
from textblob import TextBlob
import contextlib
import wave
import math
import scipy.io.wavfile
import matplotlib.pyplot as plt
import moviepy.editor
import os
from googletrans import Translator
import librosa
mysp=__import__("my-voice-analysis")
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from pydub import AudioSegment


# Security
#passlib,hashlib,bcrypt,scrypt

def make_hashes(password):
	return password

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management

conn = sqlite3.connect('data.db')
c = conn.cursor()
apikey = 'OoDWxD9rNgjw-79bjhr1MsVGcuqa8kU9OeiJbaJQlaWI'
url = 'https://api.us-south.tone-analyzer.watson.cloud.ibm.com/instances/998709a9-2c05-4199-b160-b9cf027a7425'
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data
def sppech():
        
        
        r = sr.Recognizer()

        audio = 's.wav' 
                        

        with sr.AudioFile(audio) as source:
            audio = r.record(source)
            st.write('Done!')

        try:
            text = r.recognize_google(audio,language='en-IN')
            st.write(text)


        except Exception as e:
            print(e)
        with open('text.txt', 'w') as f:
            word = text
            f.write(word + '\n')

def audio():
        audio = 'sss.wav'
        paralleldots.set_api_key("8DhrXaaW5mRir7398Ut0hmvYElXfREMtpF4ovagK0wY")
        response = paralleldots.emotion(audio)
        st.write(response)

def translator():
    q=st.text_area("enter")
    z=st.text_input("code")
    if st.button("Translate"):
            s=TextBlob(q)
            a=s.translate(from_lang=z,to='en')
            st.write(a)
def translation(c):
  translator = Translator()
  ch=translator.translate(c).text
  lang=translator.translate(c).src
  st.write(ch,lang)



def plot_spectrogram():
    """Compute power spectrogram with Short-Time Fourier Transform and plot result."""
    violin_sound_file = r"sample.wav"
    my_expander3 = st.beta_expander("PLOT SPECTOGRAM",expanded=False)
    violin_c4, _ = librosa.load(violin_sound_file)
    ipd.Audio(violin_sound_file)
    spectrogram = librosa.amplitude_to_db(librosa.stft(violin_c4))
    plt.figure(figsize=(20, 15))
    librosa.display.specshow(spectrogram, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-frequency power spectrogram for audio")
    plt.xlabel("Time")
    plt.savefig('LOGFPS.png')
    my_expander3.image('LOGFPS.png')
    X = np.fft.fft(violin_c4)
    X_mag = np.absolute(X)
    f = np.linspace(0, _, len(X_mag))
    plt.figure(figsize=(18, 10))
    plt.plot(f, X_mag) # magnitude spectrum
    plt.xlabel('Frequency (Hz)')
    plt.title("Frequency")
    plt.savefig('Freq.png')
    my_expander3.image('Freq.png')
    my_expander3.write("Length of the Frequency")
    my_expander3.write(len(violin_c4))

def wit():
	#Loading Audio Files
	mala_file = r"sample.wav"
	my_expander4 = st.beta_expander("ZCR AND RMSE",expanded=False)
	ipd.Audio(mala_file)

	# load audio files with librosa

	mala, sr = librosa.load(mala_file)

	#Root-mean-squared energy with Librosa
	FRAME_SIZE = 1024
	HOP_LENGTH = 512
	rms_mala = librosa.feature.rms(mala, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
	#Visualise RMSE + waveform
	frames = range(len(rms_mala))
	t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
	# rms energy is graphed in red

	plt.figure(figsize=(15, 17))



	plt.subplot(3, 1, 2)
	librosa.display.waveplot(mala, alpha=0.5)
	plt.plot(t, rms_mala, color="r")
	plt.ylim((-1, 1))
	plt.title("RMSE + WAVEFORM")
	plt.savefig('RMSEWAV.png')
	my_expander4.image('RMSEWAV.png')





	#Zero-crossing rate with Librosa


	zcr_mala = librosa.feature.zero_crossing_rate(mala, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
	my_expander4.write("size of ZCR")
	my_expander4.write(zcr_mala.size)



	#Visualise zero-crossing rate with Librosa

	plt.figure(figsize=(15, 10))


	plt.plot(t, zcr_mala, color="r")
	plt.title("ZCR")

	plt.ylim(0, 1)
	plt.savefig('ZCR.png')
	my_expander4.image('ZCR.png')
	#ZCR: Voice vs Noise

	voice_file = r"sample.wav"
	noise_file = r"sample.wav"

	ipd.Audio(voice_file)

	ipd.Audio(noise_file)

	# load audio files
	voice, _ = librosa.load(voice_file, duration=15)
	noise, _ = librosa.load(noise_file, duration=15)

	# get ZCR
	zcr_voice = librosa.feature.zero_crossing_rate(voice, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
	zcr_noise = librosa.feature.zero_crossing_rate(noise, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

	frames = range(len(zcr_voice))
	t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
	frames1 = range(len(zcr_noise))
	t1 = librosa.frames_to_time(frames1, hop_length=HOP_LENGTH)

	plt.figure(figsize=(15, 10))

	plt.plot(t, zcr_voice, color="r")
	plt.plot(t1, zcr_noise, color="y")
	plt.title("ZCR VOICE AND NOISE")
	plt.ylim(0, 1)
	plt.savefig('ZCRVAN.png')
	my_expander4.image('ZCRVAN.png')


def audio1():
	p="sample" # Audio File title
	my_expander6 = st.beta_expander("FEATURES",expanded=False)
	c=r"C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream" # Path to the Audio_File directory (Python 3.7)
	my_expander6.write ("number_ of_syllables=")
	my_expander6.write (mysp.myspsyl(p,c))
	my_expander6.write ("number_of_pauses=")
	my_expander6.write (mysp.mysppaus(p,c))
	my_expander6.write ("rate_of_speech=")
	my_expander6.write (mysp.myspsr(p,c))
	my_expander6.write ("articulation_rate=")
	my_expander6.write (mysp.myspatc(p,c))
	my_expander6.write ("speaking_duration=")
	my_expander6.write (mysp.myspst(p,c))
	my_expander6.write ("original_duration=")
	my_expander6.write (mysp.myspod(p,c))
	my_expander6.write ("balance=")
	my_expander6.write (mysp.myspbala(p,c))
	my_expander6.write ("f0_mean=")
	my_expander6.write (mysp.myspf0mean(p,c))
	my_expander6.write ("f0_SD=")
	my_expander6.write (mysp.myspf0sd(p,c))
	my_expander6.write ("f0_MD=")
	my_expander6.write (mysp.myspf0med(p,c))
	my_expander6.write ("f0_min=")
	my_expander6.write (mysp.myspf0min(p,c))
	my_expander6.write ("f0_max=")
	my_expander6.write (mysp.myspf0max(p,c))
	my_expander6.write ("f0_quan25=")
	my_expander6.write (mysp.myspf0q25(p,c))
	my_expander6.write ("f0_quan75=")
	my_expander6.write (mysp.myspf0q75(p,c))
	my_expander6.write (mysp.mysptotal(p,c))
	my_expander6.write ("Pronunciation_posteriori_probability_score_percentage= :%.2f"% (mysp.mysppron(p,c)))
	



def mel(audio):
	        scale_file=audio
	        my_expander2 = st.beta_expander("MEL FREQUENCY",expanded=False)
	        ipd.Audio(scale_file)
	        scale, sr = librosa.load(scale_file)
	        filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
	        my_expander2.write(filter_banks.shape)
	        plt.figure(figsize=(25, 10))
	        librosa.display.specshow(filter_banks,sr=sr,x_axis="linear") 
	        plt.colorbar(format="%+2.f")
	        plt.savefig('123.png')
	        my_expander2.image('123.png')
	        mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
	        my_expander2.write(mel_spectrogram.shape)
	        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
	        my_expander2.write(log_mel_spectrogram.shape)
	        plt.figure(figsize=(25, 10))
	        librosa.display.specshow(log_mel_spectrogram,x_axis="time",y_axis="mel",sr=sr)
	        plt.colorbar(format="%+2.f")
	        plt.savefig('1234.png')
	        my_expander2.image('1234.png')




def features():
		audio = st.file_uploader("Select file from your directory")
		if audio is not None:
			file_details = {"FileName":audio.name,"FileType":audio.type,"FileSize":audio.size}
			st.write(file_details)
			a=audio.read()
			st.audio(audio, format='audio/wav')
			rate,data=scipy.io.wavfile.read(audio)
			print(rate)
			print(data)
			plt.plot(data)
			plt.savefig('graph.png')
			plt.show()
			st.image('graph.png')
			
		



def video():
	menu = ["---------------------------------------------------------------SELECT THE LANGUAGE----------------------------------------------------------","en","hi","ml","ta","tl"]
	choice = st.selectbox("languages",menu)
	if choice !="---------------------------------------------------------------SELECT THE LANGUAGE----------------------------------------------------------":
		video = st.file_uploader("Select file from your directory")
		
		if video is not None:
			if video.type=="video/mp4":
				my_expander = st.beta_expander("VIDEO CONVERSION",expanded=False)
				my_expander1 = st.beta_expander("VIDEO ANALYSIS",expanded=False)
				file_details = {"FileName":video.name,"FileType":video.type,"FileSize":video.size}
				my_expander.write(file_details)
				a=video.read()
				my_expander.video(video, format='video/mp4')
				my_expander.write("done")
				video1 = moviepy.editor.VideoFileClip(video.name)
				audio = video1.audio
				audio.write_audiofile(r"C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream\sample.wav")
				my_expander.audio("sample.wav", format='audio/wav')
				audio1();
				audio='sample.wav'
				rate,data=scipy.io.wavfile.read('sample.wav')
				my_expander.write("RATE OF AUDIO")
				my_expander1.write(rate)
				my_expander1.write(data)
				plt.plot(data)
				plt.savefig('graph2.png')
				my_expander1.image('graph2.png')
				mel(audio);
				plot_spectrogram();
				wit();



				r = sp.Recognizer()
				with sp.AudioFile(audio) as source:
					audio = r.record(source)
					print('Done!')
				
				with st.beta_expander("TEXT AND TONE",expanded=False):
						try:
						    text1 = r.recognize_google(audio,language=choice)
						    translator = Translator()
						    ch=translator.translate(text1).text
						    lang=translator.translate(text1).src
						    st.write(ch)
						    st.success(lang)

					    #audio = 'mala.wav' \


					    

						    st.write(text)


						except Exception as e:
						    print(e)	



						
						pas=(mysp.myspgend("sample",r"C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream"))
						if pas[2]==0:
							if pas[3]==1:
								st.write("a male, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % pas[0] ,pas[1])
							elif pas[3]==2:
								st.write("a male, mood of speech: Reading, p-value/sample size= :%.2f" % pas[0], pas[1])
							else:
								st.write("a male, mood of speech: speaking passionately, p-value/sample size= :%.2f" % pas[0], pas[1])
						else:
							if pas[3]==1:
								st.write("a female, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % pas[0] ,pas[1])
							elif pas[3]==2:
								st.write("a female, mood of speech: Reading, p-value/sample size= :%.2f" % pas[0], pas[1])
							else:
								st.write("a female, mood of speech: speaking passionately, p-value/sample size= :%.2f" % pas[0], pas[1])	

						authenticator = IAMAuthenticator(apikey)
						ta = ToneAnalyzerV3(version='2017-09-21', authenticator=authenticator)
						ta.set_service_url(url)
						res = ta.tone(ch).get_result()
						st.write(res)



			else:
				audio=video
				my_expander = st.beta_expander("AUDIO ",expanded=False)
				my_expander1 = st.beta_expander("AUDIO ANALYSIS",expanded=False)
				file_details = {"FileName":audio.name,"FileType":audio.type,"FileSize":audio.size}
				my_expander.write(file_details)
				a=audio.read()
				my_expander.audio(audio, format='audio/wav')
				rate,data=scipy.io.wavfile.read(audio)
				my_expander1.write(rate)
				my_expander1.write(data)
				plt.plot(data)
				plt.savefig('graph.png')
				
				my_expander1.image('graph.png')

				my_expander12 = st.beta_expander("MEL FREQUENCY",expanded=False)
				scale_file=video
				ipd.Audio(scale_file.name)
				scale, sr = librosa.load(scale_file)
				filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
				my_expander12.write(filter_banks.shape)
				plt.figure(figsize=(25, 10))
				librosa.display.specshow(filter_banks,sr=sr,x_axis="linear")
				plt.colorbar(format="%+2.f")
				plt.savefig('123.png')
				my_expander12.image('123.png')
				mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
				my_expander12.write(mel_spectrogram.shape)
				log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
				my_expander12.write(log_mel_spectrogram.shape)
				plt.figure(figsize=(25, 10))
				librosa.display.specshow(log_mel_spectrogram,x_axis="time",y_axis="mel",sr=sr)
				plt.colorbar(format="%+2.f")
				plt.savefig('1234.png')
				my_expander12.image('1234.png')

				file_var = AudioSegment.from_ogg(video.name) 
				file_var.export('sample.wav', format='wav')
				audio1();
				plot_spectrogram();
				wit();


				
				r = sp.Recognizer()
				
				
				
				with sp.AudioFile(audio.name) as source:
					audio = r.record(source)
					print('Done!')

				with st.beta_expander("TEXT AND TONE"):
						try:
						    text1 = r.recognize_google(audio,language=choice)
						    translator = Translator()
						    ch=translator.translate(text1).text
						    lang=translator.translate(text1).src
						    st.write(ch)
						    st.success(lang)

					    #audio = 'mala.wav' \


					    

						    st.write(text)


						except Exception as e:
						    print(e)	



						
						pas=(mysp.myspgend("sample",r"C:\Users\uvima\Desktop\Streamlit app\Sentimental Analysis final\Stream"))
						if pas[2]==0:
							if pas[3]==1:
								st.write("a male, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % pas[0] ,pas[1])
							elif pas[3]==2:
								st.write("a male, mood of speech: Reading, p-value/sample size= :%.2f" % pas[0], pas[1])
							else:
								st.write("a male, mood of speech: speaking passionately, p-value/sample size= :%.2f" % pas[0], pas[1])
						else:
							if pas[3]==1:
								st.write("a female, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % pas[0] ,pas[1])
							elif pas[3]==2:
								st.write("a female, mood of speech: Reading, p-value/sample size= :%.2f" % pas[0], pas[1])
							else:
								st.write("a female, mood of speech: speaking passionately, p-value/sample size= :%.2f" % pas[0], pas[1])	

						authenticator = IAMAuthenticator(apikey)
						ta = ToneAnalyzerV3(version='2017-09-21', authenticator=authenticator)
						ta.set_service_url(url)
						res = ta.tone(ch).get_result()
						st.write(res)
		



def main():
	menu = ["Login","SignUp","Admin login"]

	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")

	elif choice == "Login":
		#st.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:
                                
				st.success("YOU LOGGED IN AS {}".format(username))
				st.markdown('<h1><center style= color:#00bfff;>WELCOME</center></h1>', unsafe_allow_html=True)
				#st.title("<h1>WELCOME</h1>")
				st.write("Build with Streamlit")

				activites=["--------SELECT THE OPTION--------","VIDEO|AUDIO","TRANSLATION","ABOUT"]
				
				choices=st.sidebar.selectbox("Select Activities",activites)

				



				

				if choices=="ABOUT":
					      st.write("This is My final year project")

				if choices=="VIDEO|AUDIO":
					      st.write("Video convertion is on process...")
					      video();
       
			        
				if choices=="TRANSLATION":
					  st.write(" Translation Done!!!")				  

			else:
				st.warning("INCORRECT Username|Password")

                           
	elif choice=="Admin login":
			username = st.sidebar.text_input("User Name")
			password = st.sidebar.text_input("Password",type='password')
			if st.sidebar.checkbox("Login"):
				if username=="ad" and password=="ad":
					st.success(" LOGGED IN AS ADMIN ")
					task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
				
                                
					if task == "Add Post":
						st.subheader("Add Your Post")

					elif task == "Analytics":
						st.subheader("Analytics")
					elif task == "Profiles":
						st.subheader("User Profiles")
						user_result = view_all_users()
						clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
						st.dataframe(clean_db)
				else:
					st.warning("INCORRECT Username|Password")


	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")



if __name__ == '__main__':
	main()
