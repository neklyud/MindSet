from flask import Flask
import joblib

app = Flask(__name__)
knn = joblib.load('knn.pkl')

@app.route('/')
def hello_world():
    print(1488)
    return '<h1> Hello, cold!!!!</h1>'

@app.route('/user/<username>')
def show_user_profile(username):
    return 'User %s' % username

@app.route('/iris/<param>')
def iris(param):
    param = param.split(',')
    param = [float(num) for num in param]
    print(param)
    import numpy as np

    print('шуе')
    param = np.array(param).reshape(1, -1)
    iris_y_test = knn.predict(param)
    return str(iris_y_test)
