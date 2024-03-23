import os
import pygame
import librosa
import webbrowser
import numpy as np
import pandas as pd
import speech_recognition as sr
import tkinter as tk
import elevenlabs
import datetime
import docx

from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from tkinter import messagebox
from tqdm import tqdm
from elevenlabs.client import ElevenLabs
from elevenlabs import play

client = ElevenLabs(
  api_key="04bbe3a8b9e8a5556157f35a3e8d8780", # Defaults to ELEVEN_API_KEY
)

voice = elevenlabs.Voice(
     voice_id='XB0fDUnXU5powFXDhCwa',
     settings=elevenlabs.VoiceSettings(
          stability = 0.5,
          similarity_boost = 0.75
     )
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df_train = pd.read_csv('C:\\Users\\randy\\Downloads\\AI PROJECT\\dataset.csv')

def speak_response(text):
    audio = client.generate(
        text=text,
        voice=voice
    )
    play(audio)

def mfcc_extract(filename):
    try:
        y, sr = librosa.load(filename, sr=None)  # Adjust sr if needed
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.T  # Transpose the matrix for HMM
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def parse_audio_files(dataframe, limit=None):
    labels = []
    features = []
    filenames = []  # Store filenames for later use
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        if limit and index >= limit:
            break
        filename = os.path.join('C:\\Users\\randy\\Downloads\\AI PROJECT', row['file_name'])  # Adjust file path
        label = row['label']
        mfcc = mfcc_extract(filename)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(label)
            filenames.append(filename)
    return features, labels, filenames

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df_train, test_size=0.2, random_state=42)

# Extract MFCC features, labels, and filenames for training data
train_features, train_labels, train_filenames = parse_audio_files(train_df)

# Extract MFCC features and labels for testing data
test_features, test_labels, test_filenames = parse_audio_files(test_df)

# Train an HMM
def train_hmm(features, labels, n_components=3, n_iter=100):
    hmm_models = {}
    unique_labels = set(labels)
    for label in unique_labels:
        label_features = [feature for feature, feature_label in zip(features, labels) if feature_label == label]
        model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter)
        model.fit(np.vstack(label_features))
        hmm_models[label] = model
    return hmm_models

# Train HMM using training features and labels
hmm_models = train_hmm(train_features, train_labels)

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        clock.tick(10)

def play_audio_by_label(label):
    audio_file_path = f"{label}.wav"  # Adjust file path
    play_audio(audio_file_path)

def recognize_intent(text):
    keywords = {
        "chrome": "chrome",
        "spotify": "spotify",
        "youtube": "youtube",
        "netflix": "netflix",
        "note": "note",
        # Add more keywords as needed
    }
    for keyword, intent in keywords.items():
        if keyword in text.lower():
            return intent
    return "unknown"  # Return "unknown" if no keyword matches

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        recognized_text = recognizer.recognize_google(audio)
        print("You said:", recognized_text)
        return recognized_text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        speak_response("Sorry, I couldn't understand what you said.")
        # play_audio_by_label("unknown")
        return None
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

def open_application_link(label):
    for index, row in df_train.iterrows():
        if row['label'].lower() == label.lower():
            application_link = row['application_link']
            if application_link:
                webbrowser.open(application_link)
            else:
                print("No application link found for this label.")
            break

def recognize_speech_and_display():
    recognized_text = recognize_speech()
    if recognized_text is not None:
        # Recognize intent
        intent = recognize_intent(recognized_text)
        if intent == "note":
            take_note(recognized_text)
        elif intent != "unknown":
            open_application_link(intent)
            speak_response(f"Yes Master, launch {intent}")
            # play_audio_by_label(intent)
        else:
            play_audio_by_label(intent)  # Play "unknown" audio

def take_note(text):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    file_name = f"note_{current_time}.docx"
    doc = docx.Document()
    doc.add_paragraph(text)
    doc.save(file_name)
    print(f"Note saved as: {file_name}")
    speak_response("Note saved successfully Master.")

def start_listening():
    speak_response("Welcome to Twinkle Virtual Asisstant, How can I help you today?")
    # play_audio_by_label("GREETINGS")
    recognize_speech_and_display()

def main():
    root = tk.Tk()
    root.title("Speech Recognition")
    root.geometry("300x100")

    btn_recognize = tk.Button(root, text="Start", command=start_listening)
    btn_recognize.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
