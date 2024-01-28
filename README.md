# audio-surveillance
An audio surveillance tool to flag malevolent activities.

The user uploads a folder containing several audio files containing conversations/dialogue. The application then flags certain audio files that could contain malevolent activites regarding terrorism and crime.

The front end and UI was made using html and css and the backend uses Python with the Flask framework.

My code converts all audio files from speech to text using google' SpeechRecognition package and stores the text as text files. My code then uses the SBERT (Sentence Bidirectional Encoder Representations from Transformers) that encodes the text files as numpy vectors (closer two vectors are to eachother, the more "similar" they are semantically) and stores these values as pickle files. I then use Facebook's FAISS (Facebook AI Similarity Search) to conduct a semantic search through my vectors to detect similarities between keywords (eg. bomb threats, murder, etc.) and the text. If the similarity metric is below a certain threshold, the corrosponding audio files are flagged for potential malevolent activites.

In order to run the application, make sure that your current working directory is audio-surveillance-flask and then run "python app.py."
