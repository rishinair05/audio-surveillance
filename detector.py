import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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
