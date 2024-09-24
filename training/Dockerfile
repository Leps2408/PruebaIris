# Usar una imagen base de Python
FROM python:3.12

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios
#COPY requirements.txt requirements.txt
COPY train.py train.py

# Instalar las dependencias
RUN pip install mlflow scikit-learn

# Comando para ejecutar el script de entrenamiento
CMD ["python", "train.py"]

