#Imports
import os
import pickle
from sentence_transformers import SentenceTransformer

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