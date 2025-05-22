from flask import Flask, request, render_template
import os
from resume_parser import extract_resume_text
from matcher import get_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume = request.files['resume']
        job_desc = request.form['job_desc']

        file_path = os.path.join(UPLOAD_FOLDER, resume.filename)
        resume.save(file_path)

        resume_text = extract_resume_text(file_path)
        score = get_similarity(resume_text, job_desc)

        return render_template('index.html', score=score)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
