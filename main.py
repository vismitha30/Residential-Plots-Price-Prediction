import pandas as pd
from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
pred5=pd.read_csv("C:\\Users\\User\\Desktop\\Cleaneddata2.csv")
pipe=pickle.load(open("C:\\Users\\User\\Desktop\\prediction1.pkl",'rb'))


@app.route('/')
def index():
    locations=sorted(pred5['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    sqft=request.form.get('total_sqft')
    distance=request.form.get('distance')
    pricepersqft=request.form.get('price_per_sqft')
    print(location,sqft,distance,pricepersqft)
    input=pd.DataFrame([[location,sqft,distance,pricepersqft]],columns=['location','total_sqft','distance','price_per_sqft'])
    prediction=pipe.predict(input)[0]*100000
    return str(np.round(prediction,1))

    


if __name__=="__main__":
    app.run(debug=True,port=5001)
    
    
    
    