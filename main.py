from gtts import gTTS
import random
import pickle
import numpy as np
import json
import speech_recognition as sr
import nltk
import pyttsx3
from nltk.stem import WordNetLemmatizer

import os

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from playsound import playsound

# Add any other required imports here (e.g., tokenizer, label encoder)

# Load the model and other necessary utilities
model = load_model('solentBot_model.keras')

# %%
# Load the tokenizer and encoder from pickle files
import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('encoder.pickle', 'rb') as enc:
    encoder = pickle.load(enc)

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
context = {}  # context is used to store the context of the conversation


# create a function to handle speech recognition
def recognize_speech():
    r = sr.Recognizer()  # initialize recognizer
    with sr.Microphone() as source:  # mention source it will be either Microphone or audio files.
        print("Speak Anything, i'm listening... :")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said : {}".format(text))
            return text
        except:
            print("Sorry could not recognize what you said")
            return input("Enter your message : ")


def clean_sentence(sentence):
    words_sentence = nltk.word_tokenize(sentence)
    words_sentence = [lemmatizer.lemmatize(word) for word in words_sentence]
    return words_sentence


def words_box(sentence, show_details=True):
    # tokenize the pattern
    words_sentence = clean_sentence(sentence)
    box = [0] * len(words)
    for j in words_sentence:
        for i, word in enumerate(words):
            if word == j:
                # assign 1 if current word is in the vocabulary position
                box[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return np.array(box)


def prediction(sentence):
    # filter out predictions below a threshold
    bd = words_box(sentence, show_details=False)
    res = model.predict(np.array([bd]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    get_back_list = []
    for r in results:
        get_back_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return get_back_list


def get_reply(intents_list, intents_json):
    tag = intents_list[0]['intent']
    all_intents = intents_json['intents']
    for i in all_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    return "Sorry, I don't understand. Can you please rephrase?"


# function to speak the text
def speak(text):
    # initialize the engine
    engine = pyttsx3.init()
    # get a list of all voices
    voices = engine.getProperty('voices')
    # Print details of each voice
    for i, voice in enumerate(voices):
        print(f"Voice {i + 1}:")
        print(f" - ID: {voice.id}")
        print(f" - Name: {voice.name}")
        print(f" - Gender: {voice.gender}")
        print(f" - Languages: {voice.languages}\n")

    # Example to select the first female voice
    for voice in voices:
        if voice.name == 'Microsoft Zira Desktop - English (United States)':
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait() # blocks while processing all the currently queued commands


# Chatbot loop with speech recognition
print("Solentbot started....... Say 'stop' or type 'close' to exit.")

while True:
    message = recognize_speech()
    if message.lower() in ['stop', 'exit', 'bye', 'quit', 'end']:
        speak("Goodbye! Feel free to return anytime you have more questions.")
        print("Goodbye!")
        break

    pred = prediction(message)
    response = get_reply(pred, intents)
    print(response)
    speak(response)