from flask import Flask, jsonify,request, render_template
from app.torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
cifar_class_mappings = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4: 'Deer', 5:'Dog',
                        6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}

def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Page of the application
@app.route('/')
def homePage():
    return render_template('/ObjectPredictorUI.html', 
                            img_name="No Image Selected", 
                            show_image=False)

@app.route('/predict',methods=['POST'])
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
            data = {'prediction': prediction.item(), 'class_name': str(cifar_class_mappings[prediction.item()])}
            return render_template('/ObjectPredictorUI.html', 
                                   img_name="upload.png",
                                   show_image=True,
                                   results=data
                                   )
        except Exception as e:
            return jsonify({'error': e})