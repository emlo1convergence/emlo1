from flask import Flask, render_template, request
from models import MobileNet
import os
import sys
import pandas as pd
from math import floor

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './static/uploads/'
app.config['result_csv'] = os.path.join(app.config['UPLOAD_FOLDER'], 'prev_result.csv')

model = MobileNet()


@app.route('/')
def index():
    file_name = app.config['sample_image']
    file_infer = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    inference, confidence = model.infer(file_infer)
    confidence = floor(confidence * 10000) / 100
    result = str(inference) +", Confidence: "+str(confidence)
    return render_template('index.html', sample_result =result)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files.getlist('file')
        result_df = pd.read_csv(app.config['result_csv'])
        lst_result=[]
        show_img1 = False
        show_img2 = False
        show_img3 = False
        img1_name = ''
        img2_name = ''
        img3_name = ''
        reslt1 = ''
        reslt2 = ''
        reslt3 = ''
        count = 0
        for item in f:
            file_name = item.filename
            file_infer = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            item.save(file_infer)
            inference, confidence = model.infer(file_infer)
            confidence = floor(confidence * 10000) / 100
            lst_result.append([inference, confidence])
            count += 1
            if(count == 1):
                show_img1 = True
                img1_name = file_name
                reslt1 = inference+", Confidence: "+str(confidence)
            elif(count == 2):
                show_img2 = True
                img2_name = file_name
                reslt2 = inference+", Confidence: "+str(confidence)
            else:
                show_img3 = True
                img3_name = file_name
                reslt3 = inference+", Confidence: "+str(confidence)
        current_df = pd.DataFrame(lst_result,columns=['inference','confidence'])
        result_df = pd.concat([result_df,current_df])
        result_df.reset_index(inplace=True)
        result_df.drop(['index'],axis=1,inplace=True)
        if ('level_0' in result_df.columns):
            result_df.drop(['level_0'],axis=1,inplace=True)
        result_df.reset_index(inplace=True)
        result_df.to_csv(app.config['result_csv'],index=False)
        result_dict = get_last_result(result_df)
        return render_template('inference.html', show_image1 =show_img1,show_image2 =show_img2,show_image3 =show_img3,
                              img_name1 = img1_name,img_name2 = img2_name,img_name3 = img3_name,
                              results1 = reslt1,results2 = reslt2,results3 = reslt3,
                              results=result_dict
                              )

def get_last_result(df):
    data={}
    df = df.sort_values(["index"],ascending=False)
    filter_df = df.head(5)
    count = 0
    for item, row in filter_df.iterrows():
        if(count == 0):
            data['r1_inference'] = row['inference']
            data['r1_confidence']= str(row['confidence'])
        elif(count == 1):
            data['r2_inference'] = row['inference']
            data['r2_confidence']= str(row['confidence'])
        elif(count == 2):
            data['r3_inference'] = row['inference']
            data['r3_confidence']= str(row['confidence'])
        elif(count == 3):
            data['r4_inference'] = row['inference']
            data['r4_confidence']= str(row['confidence'])
        else:
            data['r5_inference'] = row['inference']
            data['r5_confidence']= str(row['confidence'])
        count += 1
    return data

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.config['sample_image'] = sys.argv[1]
    app.run(host='0.0.0.0', port=port, debug=True)
