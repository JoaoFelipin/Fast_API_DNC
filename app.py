import pickle 
import pandas as pd 
from flask import Flask, request

app = Flask(__name__)

@app.route('/',methods = ['GET'])

def home():
    return 'Welcome to the Medical Insurance API'

@app.route('/predict',methods = ['POST'])
def index():
    data_json = request.get_json()['data']
    df=pd.DataFrame(data_json)

    with open('models/model.pkl','rb') as model_file:
        model = pickle.load(model_file)

    prediction = model.predict(df).tolist()
    
    return prediction

if __name__ == '__main__':
    app.run()