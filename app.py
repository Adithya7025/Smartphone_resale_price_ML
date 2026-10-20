from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name('random_forest_model_3.pkl')
model = joblib.load(MODEL_PATH)

FIELDS = ['brand','model','release_year','ram_gb','storage_gb','os_type','has_5g','original_price','age_months','condition','battery_health','screen_cracked','body_damage','repair_history','water_damage','city_tier','seller_type','warranty_remaining_months','box_available','charger_available','market_demand_score']
NUMERIC_FIELDS = ['release_year','ram_gb','storage_gb','has_5g','original_price','age_months','battery_health','screen_cracked','body_damage','repair_history','water_damage','warranty_remaining_months','box_available','charger_available','market_demand_score']

@app.get('/')
def home():
    return render_template('index.html')

@app.post('/predict')
def predict():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        missing = [field for field in FIELDS if field not in data or str(data[field]).strip() == '']
        if missing:
            return jsonify(error='Missing fields: ' + ', '.join(missing)), 400
        record = {field: data[field] for field in FIELDS}
        for field in NUMERIC_FIELDS:
            record[field] = float(record[field])
        df = pd.DataFrame([record], columns=FIELDS)
        prediction = float(model.predict(df)[0])
        return jsonify(prediction=round(prediction, 2))
    except Exception as exc:
        app.logger.exception('Prediction failed')
        return jsonify(error=f'Prediction failed: {exc}'), 500

if __name__ == '__main__':
    app.run(debug=True)
