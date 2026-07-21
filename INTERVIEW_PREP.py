#AI Interview Coach
#Speech Emotion Detection (Prototype)

#LIBRARY 
pip install sounddevice numpy librosa scikit-learn


#CODE
import sounddevice as sd
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Simulate dummy model for emotion detection
def dummy_emotion_detector(audio):
    features = np.mean(audio)  # Dummy feature
    if features > 0.01:
        return "Confident"
    else:
        return "Nervous"

def record_audio(duration=3, fs=22050):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording complete.")
    audio = np.squeeze(audio)
    return audio

# Simulate
audio = record_audio()
emotion = dummy_emotion_detector(audio)
print("Detected Emotion:", emotion)


# NLP-Based Interview Question Generation (Based on JD)
#CODE 
from transformers import pipeline

# Load a question generator model
generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")

def generate_questions(text):
    highlighted = text.replace("Python", "<hl>Python<hl>")
    result = generator("generate question: " + highlighted)
    return result[0]['generated_text']

job_desc = "We are hiring a software engineer proficient in Python, data structures, and algorithms."
question = generate_questions(job_desc)
print("Generated Interview Question:", question)
