from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Configura el URI de Tracking
mlflow.set_tracking_uri("http://172.23.73.213:30000")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cargar modelo desde MLflow usando el Run ID específico
        run_id = "model"
        model_uri = f"models:/model/1"
        model = mlflow.pyfunc.load_model(model_uri)

        # Obtener datos de la solicitud
        data = request.json
        features = pd.DataFrame(data)

        # Validar que los datos son válidos
        if features.empty:
            return jsonify({"error": "No data provided"}), 400

        # Realizar la predicción
        predictions = model.predict(features)

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

