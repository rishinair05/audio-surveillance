#Imports
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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

#Opens and reads file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def convert_text_to_sbert_vector(text, model):
    return model.encode(text)

def process_text_files(input_folder, output_folder):
    #Load the SBERT model
    model = SentenceTransformer('all-mpnet-base-v2')

    #Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #Process each text file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            text = load_text_file(input_path)

            #Convert text to SBERT vector
            sbert_vector = convert_text_to_sbert_vector(text, model)

            #Serialize the vector to a pickle file
            output_file = os.path.splitext(filename)[0] + '.pkl'
            output_path = os.path.join(output_folder, output_file)

            with open(output_path, 'wb') as f:
                pickle.dump(sbert_vector, f)

            print(f"Processed and stored: {output_file}")

input_folder = 'audio_texts'
output_folder = 'audio_vectors'

process_text_files(input_folder, output_folder)

#Unpickle the pickle files
def load_sbert_vector(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

#Unpickle all the files in audio_vectors
def load_sbert_vectors_from_folder(folder_path):
    vectors = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            vector = load_sbert_vector(file_path)
            vectors.append(vector)
            filenames.append(filename)
    return np.array(vectors), filenames

#Function to create FAISS index
def create_faiss_index(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

def get_sbert_vectors_for_keywords(keywords, model):
    return model.encode(keywords)

#Load SBERT model
model = SentenceTransformer('all-mpnet-base-v2')

#Keywords
keywords = ["terrorism", "bombing", "suicide", "kill", "murder"]

#Convert keywords to SBERT vectors
keyword_vectors = get_sbert_vectors_for_keywords(keywords, model)

#Load all SBERT vectors from the folder
folder_path = 'audio_vectors'
audio_vectors, filenames = load_sbert_vectors_from_folder(folder_path)

#Create FAISS index
faiss_index = create_faiss_index(audio_vectors)

#FAISS similarity search

flagged_files = set()

for keyword_vector in keyword_vectors:
    keyword_vector = np.array(keyword_vector, dtype=np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(keyword_vector, 1)
    for i in range(len(indices)):
        if distances[i][0] < 5:
            flagged_files.add(filenames[indices[i][0]])

for file in flagged_files:
    modified_file_name = file[:-4] + ".wav"
    print(f"Flagged File: {modified_file_name}")