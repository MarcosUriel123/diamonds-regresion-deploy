from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load('modelo_diamonds.pkl')
scaler = joblib.load('scaler_diamonds.pkl')

@app.route('/')
def home():
    return '¡Hola, Mundo! API de predicción de precio de diamantes activa.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['carat'], data['x'], data['y'], data['z'],
        data['clarity'], data['cut'], data['color']
    ]])
    features_scaled = scaler.transform(features)
    prediccion = modelo.predict(features_scaled)
    return jsonify({'precio_predicho': float(prediccion[0])})

if __name__ == '__main__':
    app.run(debug=True, port=8080)