import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, abort, redirect, url_for, render_template, send_file, flash
from werkzeug.utils import secure_filename
from preprocess import load_model
from preprocess import preprocess
from predict import predict
from predict import get_score
from flask_wtf import FlaskForm
from wtforms import StringField, FileField, SubmitField
from wtforms.validators import DataRequired
import json

app = Flask(__name__)
UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_SORT_KEYS'] = True

app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        content = request.get_json()
        path2test = content['path_to_test_csv']
        path2model = content['path_to_model']
        path2ohe = content["path_to_onehotencoder"]
        path2scaler = content["path_to_scaler"]
        path2poly = content["path_to_poly"]
        scaler = load_model(path2scaler)
        poly = load_model(path2poly)
        ohe = load_model(path2ohe)
        model = load_model(path2model)
        X_test, y_test, idx = preprocess(path2test, ohe, scaler, poly)
        prediction = predict(X_test, model)
    except:
        return redirect(url_for('bad_request'))
    #return jsonify(json.loads(pd.Series(prediction, index=idx).to_json()))
    return pd.Series(prediction, index=idx).to_json()
@app.route('/badrequest400')
def bad_request():
    return abort(400)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'file uploaded'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''