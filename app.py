from flask import Flask, render_template, request
from catboost import CatBoostRegressor
import shap

app = Flask(__name__)

# Загрузка модели
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    shap_values = None
    if request.method == 'POST':
        # Получение данных из формы
        data = [float(request.form[f'feature_{i}']) for i in range(15)]
        prediction = model.predict([data])[0]
        
        explainer = shap.Explainer(model)
        shap_values = explainer([data])
        
    return render_template('index.html', prediction=prediction, shap_values=shap_values)

if __name__ == '__main__':
    app.run(debug=True)

