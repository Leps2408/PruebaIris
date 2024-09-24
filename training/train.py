from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Cargar el dataset Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Configurar el tracking URI de MLflow
mlflow.set_tracking_uri("http://172.23.73.213:30000")  # Conectar al servidor de MLflow

# Crear y entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)
signature = infer_signature(X_test, y_pred)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Iniciar un experimento con MLflow
with mlflow.start_run() as run:
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model", signature=signature)

    print(f"Modelo registrado con ID: {run.info.run_id}")
    print(f"Modelo entrenado y registrado en MLflow con accuracy: {accuracy}")

