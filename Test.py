import speech_recognition as sr

recognizer = sr.Recognizer()

recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 100

DEVICE_INDEX = 1     # 1 se start karte hain

with sr.Microphone(device_index=DEVICE_INDEX) as source:

    print("Microphone Ready...")
    recognizer.adjust_for_ambient_noise(source, duration=2)

    print("Speak NOW...")

    audio = recognizer.listen(
        source,
        timeout=10,
        phrase_time_limit=8
    )

print("Recognizing...")

try:
    text = recognizer.recognize_google(audio, language="en-IN")
    print("You said:", text)

except Exception as e:
    print(e)

with open("voice.wav", "wb") as f:
    f.write(audio.get_wav_data())

print("Saved voice.wav")