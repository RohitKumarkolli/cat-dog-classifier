from flask import Flask, render_template, request, redirect
from app.utils.predict import model_predict as predict_image
import os

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok= True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        lable, confidence = predict_image(file_path)
        display_path = f'uploads/{filename}'
        return render_template('result.html', 
                               label = lable, 
                               confidence = confidence * 100, 
                               image_path = display_path,)
    return redirect('/')

# if you want to run the app directly
# change debug to true for development
# and false for production
if __name__ == '__main__':
    app.run(debug=True)
