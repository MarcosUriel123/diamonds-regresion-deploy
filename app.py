from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load('modelo_diamonds.pkl')
scaler = joblib.load('scaler_diamonds.pkl')

cut_map = {'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4}
color_map = {'J':0, 'I':1, 'H':2, 'G':3, 'F':4, 'E':5, 'D':6}
clarity_map = {'I1':0, 'SI2':1, 'SI1':2, 'VS2':3, 'VS1':4, 'VVS2':5, 'VVS1':6, 'IF':7}

FORM_HTML = '''
<style>
  body { font-family: Arial, sans-serif; max-width: 450px; }
  label { font-weight: bold; }
  ul { margin: 3px 0 8px 20px; padding: 0; font-size: 0.85em; }
  select, input { display: block; margin-bottom: 10px; padding: 4px; }
  h2 { margin-bottom: 5px; }
  small { color: #666; display: block; margin-top: -8px; margin-bottom: 10px; }
</style>
<h2>Predicción de Precio de Diamantes</h2>
<form method="POST" action="/predict_form">
  <label>Carat (peso del diamante)</label>
  <input type="number" step="0.01" name="carat" placeholder="ej. 0.5" required>
  <small>rango típico: 0.2 - 3.0 carat</small>

  <label>Largo - X (mm)</label>
  <input type="number" step="0.01" name="x" placeholder="ej. 5.1" required>
  <small>rango típico: 4 - 9 mm</small>

  <label>Ancho - Y (mm)</label>
  <input type="number" step="0.01" name="y" placeholder="ej. 5.2" required>
  <small>rango típico: 4 - 9 mm</small>

  <label>Profundidad - Z (mm)</label>
  <input type="number" step="0.01" name="z" placeholder="ej. 3.2" required>
  <small>rango típico: 2.5 - 5.5 mm</small>

  <label>Cut</label>
  <ul>
    <li>Ideal, Premium = mejor corte</li>
    <li>Very Good, Good = corte intermedio</li>
    <li>Fair = peor corte</li>
  </ul>
  <select name="cut">{% for c in cuts %}<option value="{{c}}">{{c}}</option>{% endfor %}</select>

  <label>Color</label>
  <ul>
    <li>D, E, F = incoloro (mejor)</li>
    <li>G, H, I = casi incoloro</li>
    <li>J = con tinte amarillo (peor)</li>
  </ul>
  <select name="color">{% for c in colors %}<option value="{{c}}">{{c}}</option>{% endfor %}</select>

  <label>Clarity</label>
  <ul>
    <li>IF, VVS1, VVS2 = sin imperfecciones (mejor)</li>
    <li>VS1, VS2 = imperfecciones muy ligeras</li>
    <li>SI1, SI2 = imperfecciones ligeras</li>
    <li>I1 = imperfecciones visibles (peor)</li>
  </ul>
  <select name="clarity">{% for c in claritys %}<option value="{{c}}">{{c}}</option>{% endfor %}</select>

  <input type="submit" value="Predecir">
</form>
{% if resultado %}<h3>Precio predicho: ${{ resultado }}</h3>{% endif %}
'''

@app.route('/')
def home():
    return render_template_string(FORM_HTML, cuts=cut_map.keys(), colors=color_map.keys(), claritys=clarity_map.keys(), resultado=None)

@app.route('/predict_form', methods=['POST'])
def predict_form():
    carat = float(request.form['carat'])
    x = float(request.form['x'])
    y = float(request.form['y'])
    z = float(request.form['z'])
    cut = cut_map[request.form['cut']]
    color = color_map[request.form['color']]
    clarity = clarity_map[request.form['clarity']]

    features = np.array([[carat, x, y, z, clarity, cut, color]])
    features_scaled = scaler.transform(features)
    prediccion = modelo.predict(features_scaled)

    return render_template_string(FORM_HTML, cuts=cut_map.keys(), colors=color_map.keys(), claritys=clarity_map.keys(), resultado=round(float(prediccion[0]), 2))

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