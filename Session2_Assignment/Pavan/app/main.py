from flask import Flask, request, jsonify, render_template, send_from_directory
from app.torch_utils import get_prediction, transform_image
from werkzeug.utils import secure_filename
import os
from copy import copy, deepcopy
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = "app/static/uploads/"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
   return render_template('index.html')

# @app.route('/show/<filename>')
# def uploaded_file(filename):
#     filename = 'http://127.0.0.1:5000/uploads/' + filename
#     return render_template('template.html', filename=filename)

# @app.route('/uploads/<filename>')
# def send_file(filename):
#     return send_from_directory("static\\uploads", filename)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        # file1 = copy(file)
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            # print("************", file.filename)
            # filename = 'http://127.0.0.1:5000/uploads/' + file.filename
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            # img_bytes = file.read()
            img_bytes = plt.imread(os.path.join(UPLOAD_FOLDER, filename)) 
            # file = request.files.get('file')
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            cifar_class_mappings = {0:'Airplane', 1:'Automobile', 2:'Bird', 
                    3:'Cat', 4: 'Deer', 5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}
            data = {'class_name': cifar_class_mappings[int(prediction.item())]}
            # return jsonify(data)
            return render_template('/output.html',
                                   img_name=file.filename,
                                   show_image=True,
                                   results=data
                                   )
        except Exception as e:
            return jsonify({'error': e})
