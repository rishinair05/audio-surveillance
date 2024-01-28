#Imports
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

#Initialize recognizer
r = sr.Recognizer()

#Text to speech with input audiofile
def transcribe_audio(path):
     # use the audio file as the audio source
    with sr.AudioFile(path) as source:
        audio_listened = r.record(source)
        # try converting it to text
        try:
            text = r.recognize_google(audio_listened)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error from Google Speech Recognition service: {e}"
    return text

# Split large audio file into several chunks when it detects gaps of silence and then transcribe
def transcribe_audio_chunks(path):
    sound = AudioSegment.from_file(path)  
    #Split into chunks based on some variables (500ms of silence and 14 decibel threshold)
    chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=sound.dBFS-14, keep_silence=500)
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        #Folder to store all audio chunks
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        #Run transcribe_audio
        try:
            text = transcribe_audio(chunk_filename)
        except sr.UnknownValueError as e:
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(chunk_filename, ":", text)
            whole_text += text

    #Save whole_text to a file
    text_folder_name = "audio_texts"
    if not os.path.isdir(text_folder_name):
        os.mkdir(text_folder_name)

    #Extracting base name from the path,
    base_name = os.path.basename(path)
    text_file_name = os.path.splitext(base_name)[0] + ".txt"
    text_file_path = os.path.join(text_folder_name, text_file_name)

    #Write to the file
    with open(text_file_path, "w", encoding="utf-8") as file:
        file.write(whole_text)
    
    print(f"Transcription saved in: {text_file_path}")
    return whole_text

#Transcribe all audio using transcribe_audio_chunks in a single directory
def transcribe_directory(directory):
    all_texts = ""
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            all_texts += transcribe_audio_chunks(file_path) + " "
    return all_texts

directory_path = "test_audios"
print("\nFull transcribed text from all files:", transcribe_directory(directory_path))