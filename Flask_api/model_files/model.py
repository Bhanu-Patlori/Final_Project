# ml_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
   
    df = df.drop_duplicates()
    df['MCV_MCH_Interaction'] = df['MCV'] * df['MCH']
    df['Log_MCHC'] = np.log(df['MCHC'] + 1)  
    features = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'MCV_MCH_Interaction', 'Log_MCHC']

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def load_and_predict(data, model):
   
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data
    processed_df = preprocess_data(df)
    y_pred = model.predict(processed_df)
    return y_pred
