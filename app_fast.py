import pickle 
import uvicorn
import pandas as pd 
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('models/model.pkl','rb') as model_file:
        model = pickle.load(model_file)

@app.get('/')
def home():
        return 'Welcome'

@app.get('/predict')
def predict(age:int,bmi:float,children:int,smoker:str):
        df_input = pd.DataFrame([dict(age=age,bmi=bmi,children=children,smoker=smoker)])
        output = model.predict(df_input)[0]
    
        return output

class Customer(BaseModel):
        age:int
        bmi:float
        children:int
        smoker: str
        class config:
                schema_extra = {
                        'example':{
                                'age':20,
                                'bmi':30.4,
                                'children':1,
                                'smoker':'no'

                        }
                }
        
@app.post('/predict_with_json')
def predict(data:Customer):
       df_input = pd.date_range([data.dict()])
       output = model.predict(df_input)[0]
       return output



if __name__ == '__main__':
        uvicorn.run(app)

