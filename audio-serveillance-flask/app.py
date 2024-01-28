from flask import Flask, render_template, request, redirect, url_for
import os
import analysis  # Importing your analysis module

app = Flask(__name__)

@app.route('/')
def index():
    result = request.args.get('result')
    return render_template('index.html', result=result)  # Pass 'None' as default

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('files')  # Get a list of files
    if not files:
        return redirect(request.url)

    upload_folder = 'uploaded_files'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    for file in files:
        if file.filename == '':
            continue
        # Save the file directly in the upload_folder
        file_path = os.path.join(upload_folder, os.path.basename(file.filename))
        file.save(file_path)

    # Call the analysis function
    result = analysis.analyze_audio(upload_folder)  # Pass the directory path

    # Redirect to the index with the result
    return redirect(url_for('index', result=result))


if __name__ == '__main__':
    app.run(debug=True)
