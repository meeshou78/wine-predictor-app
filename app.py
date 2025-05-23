from flask import Flask, request, render_template_string
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("wine_model.pkl", "rb"))

# HTML form
form = """
<!DOCTYPE html>
<html>
  <body>
    <h2>Wine Quality Prediction</h2>
    <form method="post">
      {% for feature in features %}
        {{ feature }}: <input type="number" step="any" name="{{ feature }}"><br><br>
      {% endfor %}
      <input type="submit" value="Predict">
    </form>
    {% if prediction %}
      <h3>Predicted Quality: {{ prediction }}</h3>
    {% endif %}
  </body>
</html>
"""

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        input_data = [float(request.form[f]) for f in features]
        prediction = model.predict([input_data])[0]
    return render_template_string(form, features=features, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
