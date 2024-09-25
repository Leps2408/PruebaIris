FROM python:3.12

WORKDIR /app

# Instala MLflow
RUN pip install mlflow boto3

# Crea el directorio para MLflow
RUN mkdir -p /mlflow/mlruns

# Variables de entorno para MinIO (Simulan AWS S3)
ENV MLFLOW_S3_ENDPOINT_URL="http://172.23.73.213:9000"
ENV AWS_ACCESS_KEY_ID="mi_usuario"  
ENV AWS_SECRET_ACCESS_KEY="mi_contraseña_segura"

# Define la ubicación predeterminada de los artefactos
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "s3://mlflow-artifacts"]



#FROM python:3.12

#WORKDIR /app

#RUN pip install mlflow

# Crea el directorio para MLflow
#RUN mkdir -p /mlflow/mlruns

# Define la variable de entorno para la ubicación del artefacto de MLflow
#ENV MLFLOW_ARTIFACT_URI=/mlflow/mlruns

##CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
#CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/mlruns"]
