from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

model_path = (r'C:\Users\HP\Desktop\Digi-crome\DigiCrome_Capstone_Project-1\Gradient_Boosting_Regressor.pkl')

model = pickle.load(open(model_path,'rb'))

app = Flask(__name__)

from pathlib import Path
file_path = Path(r'C:\Users\HP\Desktop\Digi-crome\DigiCrome_Capstone_Project-1\Data\feature_Property_data.csv')
data = pd.read_csv(file_path)

@app.route('/')
def index():
    Property_ID = sorted(data['PropertyID'].unique())
    return  render_template('index.html',Property_ID=Property_ID)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    Property_ID = request.form['PropertyID']
    Property_Class = request.form['PropertyClass']
    Property_Frontage = request.form['PropertyFrontage']
    Property_Size = request.form['PropertySize']
    arr = np.array([Property_ID, Property_Class, Property_Frontage, Property_Size])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))

if __name__ == '__main__':
    app.run(debug=True,port=5001)