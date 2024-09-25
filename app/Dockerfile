# Usar una imagen base de Python
FROM python:3.12

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios
#COPY requirements.txt requirements.txt
COPY app.py app.py

# Instalar las dependencias
RUN pip install Flask mlflow scikit-learn pandas
RUN pip install boto3

# Exponer el puerto 5000
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["python", "app.py"]

