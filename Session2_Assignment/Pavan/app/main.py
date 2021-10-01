from flask import Flask, request, jsonify, render_template
from app.torch_utils import get_prediction, transform_image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            cifar_class_mappings = {0:'Airplane', 1:'Automobile', 2:'Bird', 
                    3:'Cat', 4: 'Deer', 5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}
            data = {'class_name': cifar_class_mappings[int(prediction.item())]}
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': e})