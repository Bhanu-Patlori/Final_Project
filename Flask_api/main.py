import pickle
from flask import Flask, request, jsonify
from model_files.model import  load_and_predict
import numpy as np

app = Flask('app')

#To convert numpy types to Python types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() 
    else:
        return obj

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
   
    with open('./model_files/random_forest_model.pkl', 'rb') as f_in:
        model = pickle.load(f_in)
    
    predictions = load_and_predict(data, model)
    predictions = [convert_numpy(pred) for pred in predictions]

    result = {
        'rfc_prediction': predictions
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)